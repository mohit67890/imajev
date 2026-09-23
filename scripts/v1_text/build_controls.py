"""Deterministically add irrelevant images/state while retaining paired controls."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from .common import write_jsonl

NOISE_STATES=[{"ticket":{"subject":"Charged twice","priority":"normal"}},
              {"listing":{"category":"home","status":"draft"}},
              "Background workflow context: retry 0; queue intake."]

def rank(seed,*parts):return int(hashlib.sha256((seed+"\0"+"\0".join(map(str,parts))).encode()).hexdigest()[:16],16)

def add_controls(text_rows,image_rows,seed="decision-v1-text",text_image_rate=.125,image_state_rate=.15):
    image_partitions={}
    for row in image_rows:
        for image in row.get('images',[]):image_partitions.setdefault(image.get('sha256',image.get('image')),set()).add(row.get('partition'))
    relevant=[r for r in image_rows if r.get("images") and r.get("partition")=="train"
              and all(image_partitions.get(im.get('sha256',im.get('image')))=={'train'} for im in r['images'])]
    if text_rows and not relevant:raise ValueError("text-image controls require training images")
    out=[json.loads(json.dumps(r)) for r in image_rows]
    # Attach only images already retained in a relevant record, and mark both roles for auditing.
    for r in out:
        if r.get("images"):r.setdefault("image_role","relevant")
        if r.get("partition")=="train" and r.get("images") and rank(seed,r["id"],"state")%10000<int(image_state_rate*10000):
            req=r.setdefault("request",{});state=req.get("state",{})
            noise=NOISE_STATES[rank(seed,r["id"],"which")%len(NOISE_STATES)]
            # Preserve label-bearing state. A reserved namespace makes the injected
            # context explicitly additive and avoids overwriting cited keys.
            if isinstance(state,dict):
                if "_irrelevant_context" in state:continue
                state=json.loads(json.dumps(state));state["_irrelevant_context"]=noise;req["state"]=state;r["state_role"]="mixed_relevant_and_irrelevant" if len(state)>1 else "irrelevant"
    for r0 in text_rows:
        r=json.loads(json.dumps(r0))
        if r.get("partition")=="train" and rank(seed,r["id"],"image")%10000<int(text_image_rate*10000):
            donor=relevant[rank(seed,r["id"],"donor")%len(relevant)];r["images"]=json.loads(json.dumps(donor["images"]));r["image_role"]="irrelevant";r["irrelevant_image_source_id"]=donor["id"]
        out.append(r)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("text",type=Path);p.add_argument("images",type=Path);p.add_argument("output",type=Path);p.add_argument("--seed",default="decision-v1-text");a=p.parse_args()
    load=lambda x:[json.loads(s) for s in x.open() if s.strip()]
    rows=add_controls(load(a.text),load(a.images),a.seed);write_jsonl(a.output,rows);print(json.dumps({"records":len(rows),"output":str(a.output)}))
if __name__=="__main__":main()


def grounded_contradictions(image_rows, limit=2000, seed='decision-v1-text'):
    """Use known ABO category annotations to contrast photo and listing evidence."""
    partitions={}
    for row in image_rows:
        for image in row.get('images',[]):partitions.setdefault(image.get('sha256',image.get('image')),set()).add(row.get('partition'))
    eligible=[r for r in image_rows if r.get('source')=='abo' and r.get('partition')=='train'
              and r.get('images') and str(r.get('source_answer','')).startswith('category=')
              and r.get('abstention_cause') is None
              and all(partitions.get(im.get('sha256',im.get('image')))=={'train'} for im in r['images'])]
    categories=sorted({r['source_answer'].split('=',1)[1] for r in eligible})
    if len(categories)<2:return []
    out=[];seen=set()
    for donor in sorted(eligible,key=lambda r:rank(seed,r['id'],'contradiction')):
        sha=donor['images'][0].get('sha256')
        if sha in seen:continue
        seen.add(sha)
        actual=donor['source_answer'].split('=',1)[1]
        alternatives=[x for x in categories if x!=actual]
        listed=alternatives[rank(seed,donor['id'])%len(alternatives)]
        values=[actual,listed]
        if rank(seed,donor['id'],'order')%2:values.reverse()
        copy=json.loads(json.dumps(donor));copy['id']+=':evidence-contradiction';copy['family']='photo_listing_contradiction'
        copy['request']={'schema_version':'1.0','request_id':copy['id'],'state':{'listing':{'category':listed}},'fields':[
            {'id':'photo','type':'choice','question':'According to the photo, which product category is shown? Ignore the listing when answering this question.','options':[{'value':v} for v in values]},
            {'id':'listing','type':'choice','question':'According to listing.category, which category is recorded? Answer from the listing even if the photo differs.','options':[{'value':v} for v in values]}]}
        copy.pop('target',None);copy.pop('target_distribution',None);copy['targets']={'photo':actual,'listing':listed}
        copy['image_role']='relevant';copy['control_kind']='evidence_source_contradiction';copy['parent_id']=donor['id']
        out.append(copy)
        if len(out)>=limit:break
    return out
