"""PyTorch evaluation of a decision manifest's test partition: served single pass, batched by length.

A fraction of multiple-choice cases is also scored with reversed options to measure order sensitivity.
Batched scoring needs CUDA (left padding produces NaN on Apple MPS); use --max-batch 1 elsewhere.
"""
import json,time,argparse,hashlib
from pathlib import Path
import torch
from vision_decision.contracts import UNKNOWN
from decision_data import load_records,load_image,render,pixel_budget,approx_tokens,VERSION
from torch_decision import TorchDecision

parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--model',required=True);parser.add_argument('--revision');parser.add_argument('--adapter')
parser.add_argument('--device',default='cuda');parser.add_argument('--version',default=VERSION);parser.add_argument('--pixels',type=int,default=400000);parser.add_argument('--limit',type=int,default=0)
parser.add_argument('--partition',choices=['train','dev','calibration','test'],default='test')
parser.add_argument('--shard',default='0/1',help='i/n: evaluate every n-th case starting at i');parser.add_argument('--token-budget',type=int,default=16000);parser.add_argument('--max-batch',type=int,default=48)
parser.add_argument('--workers',type=int,default=8);parser.add_argument('--order-check',type=float,default=.25,help='share of multiple-choice cases also scored with reversed options');args=parser.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True);rows=load_records(args.partition,args.version)
if args.limit:rows=rows[::max(1,len(rows)//args.limit)][:args.limit]
shard,shards=map(int,args.shard.split('/'));rows=rows[shard::shards]
path=args.model
if not Path(path).is_dir():
 from huggingface_hub import snapshot_download
 path=snapshot_download(args.model,revision=args.revision)
engine=TorchDecision(path,args.device)
if args.adapter:
 from peft import PeftModel
 engine.model=PeftModel.from_pretrained(engine.model,args.adapter).eval()
 engine.enable_readout(args.adapter,trainable=False)  # False return preserves legacy adapter scoring
name=lambda v:'unknown' if v==UNKNOWN or v is None else str(v).lower() if isinstance(v,bool) else v
def reversed_copy(r):
 copy=json.loads(json.dumps(r));copy['request']['fields'][0]['options'].reverse();return copy
# work items: (row index, is_reversed); reversed twins only for a deterministic share of multiple-choice cases
items=[(i,False) for i in range(len(rows))]+[(i,True) for i,r in enumerate(rows) if r['request']['fields'][0]['type']=='choice' and int(hashlib.sha256(r['id'].encode()).hexdigest(),16)%1000<args.order_check*1000]
items.sort(key=lambda it:approx_tokens(rows[it[0]],args.pixels));batches=[];current=[];longest=0
for it in items:
 n=approx_tokens(rows[it[0]],args.pixels)
 if current and (max(longest,n)*(len(current)+1)>args.token_budget or len(current)>=args.max_batch):batches.append(current);current=[];longest=0
 current.append(it);longest=max(longest,n)
if current:batches.append(current)
class Work(torch.utils.data.Dataset):
 def __len__(self):return len(batches)
 def __getitem__(self,b):
  examples=[];meta=[]
  for i,rev in batches[b]:
   r=reversed_copy(rows[i]) if rev else rows[i];header,choices,texts,target=render(r);target=max(range(len(target)),key=target.__getitem__) if isinstance(target,list) else target
   labels=engine.labels(len(choices),len(r['images']));images=[load_image(x['image'],pixel_budget(r,args.pixels)) for x in r['images']]
   examples.append((*engine.render_example(images,header+'\n'.join(f'{l}: {t}' for l,t in zip(labels,texts)),labels),target));meta.append((i,rev,[name(c[0]) for c in choices],[c[0] for c in choices],target))
  return engine.collate(examples),meta
loader=torch.utils.data.DataLoader(Work(),batch_size=None,shuffle=False,num_workers=args.workers,prefetch_factor=4 if args.workers else None)
results={};reversed_prediction={};start=time.monotonic();done=0
with torch.no_grad():
 for (inputs,token_ids,_),meta in loader:
  began=time.monotonic();logits=engine.candidate_logits_batch(inputs,token_ids);each=(time.monotonic()-began)/len(meta)
  for x,(i,rev,names,choice_values,target) in zip(logits,meta):
   probs=x.softmax(0).cpu().tolist();best=max(range(len(probs)),key=probs.__getitem__)
   if rev:reversed_prediction[i]=names[best];continue
   r=rows[i];results[i]=dict(id=r['id'],group=r.get('group',r.get('source_group',r['id'])),source=r.get('source'),family=r['family'],field_type=r['request']['fields'][0]['type'],abstention_cause=r.get('abstention_cause'),options=len(names)-1,
    partition=args.partition,heldout_family=bool(r.get('heldout_family',False)),decision_type={'noul':'boolean','score':'ordinal'}.get(r['request']['fields'][0]['type'],r['request']['fields'][0]['type']),option_count=len(names)-1,
    labels=[UNKNOWN if v==UNKNOWN else name(v) for v in choice_values],logits=x.float().cpu().tolist(),target_index=target,
    target=names[target],prediction=names[best],confidence=probs[best],scores=dict(zip(names,probs)),raw_logits=dict(zip(names,x.float().cpu().tolist())),seconds=each,correct=names[best]==names[target])
  done+=len(meta)
  if done%2000<len(meta):print(done,'/',len(items),f'{done/(time.monotonic()-start):.0f} cases/s',flush=True)
for i,p in reversed_prediction.items():results[i]['reversed_prediction']=p
for i,r in enumerate(rows):
 for key in ('pair_id','control_variant'):
  if key in r:results[i][key]=r[key]
(out/(f'predictions-{shard}.jsonl' if shards>1 else 'predictions.jsonl')).write_text(''.join(json.dumps(results[i])+'\n' for i in range(len(rows))))
print('Finished evaluation:',len(rows),'cases in',round(time.monotonic()-start),'s',flush=True)
