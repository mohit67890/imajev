"""Assemble bounded, deterministic text/image training strata; retain every eval row."""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
from .build_controls import add_controls
from .common import write_jsonl

def _rank(seed,r):return hashlib.sha256(f"{seed}\0{r['id']}".encode()).hexdigest()

def stratified_train(rows,limit,seed):
    """Round-robin source/family strata under an expanded-decision, not request, cap."""
    buckets=collections.defaultdict(list);fixed=[]
    for r in rows:
        (buckets[(r.get("source"),r.get("family"))] if r.get("partition")=="train" else fixed).append(r)
    for key in buckets:buckets[key].sort(key=lambda r:_rank(seed,r))
    chosen=[];keys=sorted(buckets)
    used=0
    while used<limit and keys:
        alive=[]
        for key in keys:
            if buckets[key] and used<limit:
                row=buckets[key].pop();cost=len(row.get("request",{}).get("fields",[])) or 1
                if used+cost<=limit:chosen.append(row);used+=cost
                if buckets[key]:alive.append(key)
        keys=alive
    return fixed+chosen

def assemble(text_rows,image_rows,text_limit=200_000,image_limit=300_000,seed="decision-v1-text"):
    text=stratified_train(text_rows,text_limit,seed+":text")
    images=stratified_train(image_rows,image_limit,seed+":image")
    return add_controls(text,images,seed)

def main():
    p=argparse.ArgumentParser();p.add_argument("--text",type=Path,nargs="+",required=True);p.add_argument("--images",type=Path,nargs="+",required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--text-limit",type=int,default=200_000);p.add_argument("--image-limit",type=int,default=300_000);p.add_argument("--seed",default="decision-v1-text");a=p.parse_args()
    load=lambda paths:[json.loads(line) for path in paths for line in path.open() if line.strip()]
    rows=assemble(load(a.text),load(a.images),a.text_limit,a.image_limit,a.seed);write_jsonl(a.output,rows)
    print(json.dumps({"records":len(rows),"output":str(a.output)}))
if __name__=="__main__":main()
