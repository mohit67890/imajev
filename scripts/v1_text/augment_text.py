"""Deterministic, label-preserving request variants and grounded abstention controls."""
import copy, hashlib, json

def rank(r,suffix=''):
 return int(hashlib.sha256((r['id']+suffix).encode()).hexdigest()[:12],16)

def substantive(value):
 if isinstance(value,dict):return any(substantive(v) for v in value.values())
 if isinstance(value,list):return any(substantive(v) for v in value)
 return value not in (None,'')

def target_literal_in_question(row):
 questions=' '.join(f.get('question','') for f in row['request']['fields']).casefold()
 values=list((row.get('targets') or {}).values()) or [row.get('target')]
 return any(v is not None and len(str(v).strip())>=2 and str(v).strip().casefold() in questions for v in values)

def augment(rows):
 out=[]
 for original in rows:
  r=copy.deepcopy(original)
  if r.get('partition')!='train':out.append(r);continue
  req=r['request']
  # Enforce the planned presentation mix across sources without dropping evidence:
  # 40% plain JSON text, 60% structured JSON. Empty abstention controls stay empty.
  state=req.get('state')
  if substantive(state):
   if rank(r,'presentation')%100<40 and isinstance(state,dict):req['state']=json.dumps(state,sort_keys=True,ensure_ascii=False,separators=(',',':'))
   elif rank(r,'presentation')%100>=40 and isinstance(state,str):req['state']={'text':state}
  if rank(r,'instruction')%100<6:
   for field in req['fields']:field['question']='question: '+field['question']
   r['request_style']='object_instruction_flattened'
  if len(req['fields'])==1 and not r.get('targets') and req['fields'][0]['type']=='choice':
   f=req['fields'][0]
   if len(f['options'])<254 and not any(o['value']=='other' for o in f['options']) and rank(r,'other')%100<10:
    f['options'].append({'value':'other','description':'None of the listed answers applies.'})
    if r.get('abstention_cause')=='not_listed':r['target']='other';r['abstention_cause']=None
  out.append(r)
  # Additional examples are explicitly derived from their parent and share its group.
  if rank(r,'abstain')%100>=12:continue
  c=copy.deepcopy(r);c['id']+=':text-control';c['request']['request_id']=c['id'][:128];c['parent_id']=r['id']
  fields=c['request']['fields'];first=fields[0]
  if len(fields)==1 and not r.get('targets') and first['type']=='choice' and len(first['options'])>=3 and r.get('target') is not None and rank(r,'cause')%2:
   first['options']=[o for o in first['options'] if o['value']!=r['target'] and o['value']!='other']
   if len(first['options'])<2:continue
   c['target']=None;c['abstention_cause']='not_listed'
  else:
   # Clearing an already-empty state creates an answer-only question mislabeled unknown.
   # A literal gold answer in the question has the same defect after state removal.
   if not substantive(c['request'].get('state')) or target_literal_in_question(c):continue
   c['request']['state']={}
   for f in fields:f['question']='Using only the supplied state, answer this question; if the required evidence is absent, choose unknown. '+f['question']
   if len(fields)==1 and not c.get('targets'):c['target']=None;c['abstention_cause']='insufficient_evidence'
   else:
    c['targets']={f['id']:None for f in fields};c['abstention_causes']={f['id']:'insufficient_evidence' for f in fields}
   c['abstention_cause']='insufficient_evidence'
  c.pop('target_distribution',None);c.pop('target_distributions',None)
  out.append(c)
 return out


def reference_controls(rows, limit=1000):
 """Unknown for a nonexistent record or second item in a one-item listing."""
 out=[]
 for r in sorted(rows,key=lambda r:rank(r,'reference')):
  if r.get('source')!='abo' or r.get('partition')!='train' or len(r['request']['fields'])!=1 or r.get('target') is None:continue
  c=copy.deepcopy(r);cause='mismatched_reference' if len(out)%2 else 'false_premise'
  c['id']+=':'+cause;c['request']['request_id']=c['id'][:128];c['parent_id']=r['id'];c['target']=None;c['abstention_cause']=cause
  field=c['request']['fields'][0]
  if cause=='mismatched_reference':
   c['request']['state']={'records':{'record_A':r['request']['state']}}
   field['question']='For record_B only (not record_A), answer: '+field['question']
  else:
   c['request']['state']={'listings':[r['request']['state']]}
   field['question']='For the second listing in the supplied listings array, answer: '+field['question']
  c.pop('target_distribution',None);out.append(c)
  if len(out)>=limit:break
 return out
