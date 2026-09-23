"""Resumable local generation with structural validation and independent blind labels."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
from vision_decision.contracts import Request, UNKNOWN
from vision_decision.backend import MLXDirect
from decision_data import expand_fields, render

DOMAINS=['support tickets','invoices','inventory','calendar scheduling','shipping','travel bookings','software incident triage','document review','subscription management','expense approvals']
SHAPES=[('choice','boolean'),('ordinal','choice','boolean'),('boolean','ordinal'),('choice','choice','boolean','ordinal'),('ordinal','boolean','choice','boolean','choice')]
ROOT=Path(__file__).resolve().parents[2]

def field_instructions(kinds):
 parts=[]
 for n,kind in enumerate(kinds,1):
  fid=f'decision_{n}'
  if kind=='choice':parts.append(f'{fid}: choice with 2-4 short, unique string option values')
  elif kind=='boolean':parts.append(f'{fid}: boolean with a yes/no question')
  else:parts.append(f'{fid}: ordinal with 3-5 ascending integer levels starting at 0, each with a meaningful description')
 return '; '.join(parts)

def generation_prompt(domain,i,kinds,unknown_ids):
 missing=', '.join(unknown_ids) if unknown_ids else 'none'
 examples=[]
 for n,kind in enumerate(kinds,1):
  base={'id':f'decision_{n}','type':kind,'question':f'<question {n}>'}
  if kind=='choice':base['options']=[{'value':'route_a'},{'value':'route_b'}]
  if kind=='ordinal':base['levels']=[{'value':0,'description':'low'},{'value':1,'description':'medium'},{'value':2,'description':'high'}]
  examples.append(base)
 shape=json.dumps({'state':{'domain':domain,'facts':{'<fact_name>':'<fact_value>'},'policies':['<explicit if/then rule>'],'missing_for':unknown_ids},'fields':examples,'targets':{f'decision_{n}':None if f'decision_{n}' in unknown_ids else '<valid value of the required type>' for n in range(1,len(kinds)+1)}},ensure_ascii=False)
 return f'''Create one original, realistic decision record for the domain "{domain}", variation {i}. Return ONLY one JSON object with keys state, fields, targets.
Use exactly {len(kinds)} fields in this exact order and with these exact IDs/types: {field_instructions(kinds)}.
The `fields` value MUST be a JSON array of field objects, never an object/map. Follow this structural blueprint, replacing every angle-bracket placeholder and using correctly typed targets: {shape}
state MUST be an object with exactly these keys:
- domain: exactly "{domain}"
- facts: an object of concrete fictional observations
- policies: a nonempty list of explicit if/then rules. For every answerable choice, the matching rule must name the exact option value. Policies must give enough information to derive every non-null target without outside knowledge.
- missing_for: exactly [{', '.join(json.dumps(x) for x in unknown_ids)}]. For these fields, deliberately omit one required fact and set the target to null. Do not state or imply that fact. Every other field must be answerable and non-null.
targets must have exactly the field IDs and valid values: choice targets exactly match an option value; boolean targets are true/false; ordinal targets exactly match a listed integer level. Avoid ambiguous thresholds, unstated authorization, vague mappings, and questions requiring common-sense guesses. Make field questions materially different rather than asking the same decision twice. Keep state under 140 words and each question concise. The mechanically required missing fields are: {missing}. No markdown or commentary.'''

def validate_candidate(obj,domain,kinds,unknown_ids,row_id):
 if set(obj)!= {'state','fields','targets'}:raise ValueError('top-level keys must be exactly state, fields, targets')
 state=obj['state']
 if not isinstance(state,dict) or set(state)!= {'domain','facts','policies','missing_for'}:raise ValueError('state shape invalid')
 if state['domain']!=domain:raise ValueError('domain mismatch')
 if not isinstance(state['facts'],dict) or not state['facts']:raise ValueError('facts must be a nonempty object')
 if not isinstance(state['policies'],list) or not state['policies'] or not all(isinstance(x,str) and x.strip() for x in state['policies']):raise ValueError('policies invalid')
 if state['missing_for']!=unknown_ids:raise ValueError('missing_for does not match requested unknown fields')
 request=Request.model_validate({'schema_version':'1.0','request_id':row_id,'state':state,'fields':obj['fields']})
 expected_ids=[f'decision_{i}' for i in range(1,len(kinds)+1)]
 if [f.id for f in request.fields]!=expected_ids or [f.type for f in request.fields]!=list(kinds):raise ValueError('field IDs/types/arity mismatch')
 targets=obj['targets']
 if not isinstance(targets,dict) or list(targets)!=expected_ids:raise ValueError('targets must contain field IDs in order')
 for field in request.fields:
  target=targets[field.id];missing=field.id in unknown_ids
  if (target is None)!=missing:raise ValueError(f'{field.id}: null target does not match missing_for')
  if target is None:continue
  if field.type=='choice':
   values=[x.value for x in field.options]
   if target not in values or type(target) is not str:raise ValueError(f'{field.id}: invalid choice target')
   if target not in json.dumps(state['policies'],ensure_ascii=False):raise ValueError(f'{field.id}: choice route absent from explicit policy')
  elif field.type=='boolean':
   if type(target) is not bool:raise ValueError(f'{field.id}: boolean target must be true/false')
  elif type(target) is not int or target not in [x.value for x in field.levels]:raise ValueError(f'{field.id}: invalid ordinal target')
 candidate={'id':row_id,'request':request.model_dump(),'targets':targets}
 for item in expand_fields(candidate):render(item)
 return request,targets

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT/'artifacts/model.json');p.add_argument('--count',type=int,default=100);p.add_argument('--max-attempts',type=int,default=300);p.add_argument('--output',type=Path,default=ROOT/'data/decision-v1-text/synthetic/local-pilot.jsonl');a=p.parse_args()
 a.output.parent.mkdir(parents=True,exist_ok=True)
 existing=[json.loads(l) for l in a.output.open()] if a.output.exists() else []
 engine=MLXDirect(str(a.bundle))
 from mlx_vlm import generate
 from mlx_vlm.prompt_utils import apply_chat_template
 start=time.monotonic();kept=sum(r.get('agreement') is True for r in existing);valid=sum('request' in r for r in existing);attempts=len(existing)
 for i in range(len(existing),a.max_attempts):
  if kept>=a.count:break
  attempts=i+1;domain=DOMAINS[i%len(DOMAINS)];kinds=SHAPES[i%len(SHAPES)]
  ids=[f'decision_{n}' for n in range(1,len(kinds)+1)];unknown_ids=[ids[(i//len(SHAPES))%len(ids)]] if i%3==0 else []
  prompt=generation_prompt(domain,i,kinds,unknown_ids)
  rendered=apply_chat_template(engine.processor,engine.model.config,prompt,num_images=0,enable_thinking=False)
  raw=generate(engine.model,engine.processor,rendered,image=None,max_tokens=1600,temperature=.65,verbose=False).text
  row={'id':f'local-synthetic-v2-{i:06d}','domain':domain,'requested_types':list(kinds),'required_unknown_fields':unknown_ids,'model_revision':engine.bundle['revision'],'generation_prompt':prompt,'raw_generation':raw,'review_status':'pending_human_review','training_admitted':False}
  try:
   begin=raw.index('{');obj,_=json.JSONDecoder().raw_decode(raw[begin:]);request,teacher=validate_candidate(obj,domain,kinds,unknown_ids,row['id']);valid+=1
   blind={}
   for field in request.fields:
    response=engine.generated_reference(None,field,request.state)
    if not response['format_valid']:raise ValueError('blind label output invalid')
    blind[field.id]=None if response['value']==UNKNOWN else response['value']
   agreement=teacher==blind
   row.update(request=request.model_dump(),blind_labels=[teacher,blind],agreement=agreement)
   row['blind_input_sha256']=hashlib.sha256(json.dumps(row['request'],sort_keys=True).encode()).hexdigest();kept+=agreement
  except (ValueError,KeyError,AssertionError,TypeError,json.JSONDecodeError) as exc:row.update(agreement=False,error=str(exc)[:500])
  with a.output.open('a') as f:f.write(json.dumps(row,ensure_ascii=True,allow_nan=False)+'\n')
  print(json.dumps({'attempt':attempts,'valid':valid,'agreed':kept,'seconds':round(time.monotonic()-start,1)}),flush=True)
 report={'attempts':attempts,'valid_candidates':valid,'agreements':kept,'requested':a.count,'seconds':time.monotonic()-start,'human_review':'pending','training_admitted':False,'model_revision':engine.bundle['revision']}
 (a.output.with_suffix('.report.json')).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
