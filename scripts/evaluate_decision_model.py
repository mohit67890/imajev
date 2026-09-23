"""Stage 4: single-pass served scoring of decision-v0 test records, base model or base + adapter.

One pass in the recorded option order (what serving does) plus one pass with choice options
reversed, to measure order sensitivity. No rotations, no fitted calibrator.
"""
import os,json,time,argparse,hashlib
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
from vision_decision.backend import MLXDirect
from vision_decision.contracts import UNKNOWN
from decision_data import load_records,load_image,render,VERSION

parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--bundle',default='artifacts/model.json');parser.add_argument('--adapter')
parser.add_argument('--partition',default='test');parser.add_argument('--version',default=VERSION);parser.add_argument('--pixels',type=int,default=400000);parser.add_argument('--limit',type=int,default=0);parser.add_argument('--filter',action='append',default=[],help='key=value on the expanded record (e.g. group=held_out_source); repeatable');args=parser.parse_args()
out=Path('reports')/args.version/'eval'/args.name;out.mkdir(parents=True,exist_ok=True);rows=load_records(args.partition,args.version)
for f in args.filter:
 k,v=f.split('=',1);rows=[r for r in rows if str(r.get(k))==v]
if args.limit:rows=rows[::max(1,len(rows)//args.limit)][:args.limit]
adapter_hash=hashlib.sha256((Path(args.adapter)/'adapters.safetensors').read_bytes()).hexdigest() if args.adapter else None
config=dict(vars(args),model=json.loads(Path(args.bundle).read_text()),adapter_sha256=adapter_hash,manifest_sha256=hashlib.sha256(Path(f'data/manifests/{args.version}.jsonl').read_bytes()).hexdigest(),cases=len(rows))
p=out/'config.json'
if p.exists():assert json.loads(p.read_text())==config,'Evaluation exists with a different config; choose a new --name'
else:p.write_text(json.dumps(config,indent=2)+'\n')
p=out/'predictions.jsonl';done=list(map(json.loads,p.read_text().splitlines())) if p.exists() else [];assert [r['id'] for r in done]==[r['id'] for r in rows[:len(done)]]
backend=MLXDirect(bundle=args.bundle,adapter=args.adapter);name=lambda v:'unknown' if v==UNKNOWN or v is None else str(v).lower() if isinstance(v,bool) else v
def reversed_record(r):
 copy=json.loads(json.dumps(r));field=copy['request']['fields'][0]
 if field['type']=='choice':field['options'].reverse()
 return copy
with p.open('a') as stream:
 for i,r in enumerate(rows[len(done):],len(done)):
  image=[load_image(x['image'],args.pixels) for x in r['images']];image=image[0] if len(image)==1 else image;header,choices,texts,target=render(r)
  start=time.monotonic();results,meta=backend.score_questions(image,[(header,choices,texts)],rotations=1);seconds=time.monotonic()-start;result=results[0]
  row=dict(id=r['id'],group=r.get('group'),family=r.get('family'),source=r.get('source'),field_type=r['request']['fields'][0]['type'],abstention_cause=r['abstention_cause'],options=len(choices)-1,
   target=name(choices[target][0]),prediction=name(result.value),confidence=max(result.scores.values()),scores={name(k):v for k,v in result.scores.items()},seconds=seconds,
   prefill_seconds=meta['prefill_seconds'],input_tokens=meta['questions'][0]['rotations'][0]['input_tokens'],
   partition=r.get('partition'),decision_type=r['request']['fields'][0]['type'],option_count=len(choices)-1,target_index=target,
   labels=[name(c[0]) for c in choices],logits=[float({str(name(k)):v for k,v in result.raw_logits.items()}[str(name(c[0]))]) for c in choices] if result.raw_logits else None,
   raw_logits={str(name(k)):float(v) for k,v in result.raw_logits.items()} if result.raw_logits else None,has_image=bool(r['images']))
  row['correct']=row['prediction']==row['target']
  if r['request']['fields'][0]['type']=='choice':
   h,c,t,_=render(reversed_record(r));row['reversed_prediction']=name(backend.score_questions(image,[(h,c,t)],rotations=1)[0][0].value)
  stream.write(json.dumps(row)+'\n');stream.flush()
  if (i+1)%50==0:print(args.name,i+1,'/',len(rows),flush=True)
  backend.mx.clear_cache()
print('Finished',args.name,flush=True)
