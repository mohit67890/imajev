"""Yes/no questions whose honest answer is "unknown" — the gap the other converters left (1.7% of booleans).

TDIUC "absurd" questions ask about an object that COCO annotations prove is absent. Rephrased as a
yes/no claim ("Is the giraffe brown?") the premise is false, so the target is unknown. The same
templates are also generated from answerable TDIUC questions (true and false in equal numbers), so the
wording never gives the answer away. Reuses COCO images already fetched by other sources (hardlinks).
"""
import json,zipfile,random,hashlib,os
from pathlib import Path
from collections import Counter,defaultdict
SEED=20260922;N_UNKNOWN=16000;N_TEST=600;root=Path('data/decision-v1/bool_abstain');(root/'images').mkdir(parents=True,exist_ok=True)
excl=json.loads(Path('data/decision-v1/exclusions.json').read_text());banned_coco=set(excl['coco_image_ids']);banned_sha=set(excl['sha256'])
# COCO id -> image entry and partition side, from sources that key source_group by COCO id
onfile={};side={}
for s in ('vqav2','tdiuc','aokvqa'):
 for line in (Path('data/decision-v1')/s/'records.jsonl').read_text().splitlines():
  r=json.loads(line)
  if len(r['images'])!=1 or not str(r['source_group']).isdigit():continue
  g=int(r['source_group']);onfile.setdefault(g,r['images'][0]);side.setdefault(g,set()).add('test' if r['partition']=='test' else 'fit')
CONFUSABLE=[{'gray','silver','white'},{'brown','tan','beige','gold','orange','yellow'},{'red','pink','purple','orange'},{'blue','purple'},{'black','gray'}]
def clash(a,b):return a==b or any(a in g and b in g for g in CONFUSABLE)
PATTERNS={'What color is the ':('is','color'),'What color are the ':('are','color'),'What material is the ':('is','material'),'What material are the ':('are','material')}
PHRASES={'color':['{V} the {x} {a}?','{V} the {x} {a} in color?','Would you say the {x} {v} {a}?','Does the {x} look {a}?|Do the {x} look {a}?','{V} the color of the {x} {a}?|{V} the colors of the {x} {a}?'],
 'material':['{V} the {x} made of {a}?','{V} the {x} {a}?','Does the {x} appear to be made of {a}?|Do the {x} appear to be made of {a}?','Would you say the {x} {v} made of {a}?','{V} {a} what the {x} {v} made of?']}
def phrase(kind,verb,x,a,rng):
 t=rng.choice(PHRASES[kind]);t=t.split('|')[0 if verb=='is' else -1];return t.format(V=verb.capitalize(),v=verb,x=x,a=a)
def parse(q):
 for prefix,(verb,kind) in PATTERNS.items():
  if q.startswith(prefix) and q.endswith('?'):
   x=q[len(prefix):-1].strip()
   if 0<len(x)<=60 and ' made of' not in x:return verb,kind,x
 return None
rows=[]
with zipfile.ZipFile('.cache/datasets/tdiuc/TDIUC.zip') as z:
 for split in ('train','val'):
  ann=json.loads(z.read(f'TDIUC/Annotations/mscoco_{split}2014_annotations.json'))['annotations'];qs={q['question_id']:q['question'] for q in json.loads(z.read(f'TDIUC/Questions/OpenEnded_mscoco_{split}2014_questions.json'))['questions']}
  for a in ann:
   if a['question_type'] in ('absurd','color','attribute') and a['image_id'] in onfile:
    p=parse(qs[a['question_id']])
    if p:rows.append(dict(qid=f'{split}-{a["question_id"]}',image_id=a['image_id'],family=a['question_type'],answer=a['answers'][0]['answer'].replace('grey','gray'),verb=p[0],kind=p[1],x=p[2]))
values={k:[v for v,_ in Counter(r['answer'] for r in rows if r['family']!='absurd' and r['kind']==k).most_common(14)] for k in ('color','material')}
rng=random.Random(SEED);rows.sort(key=lambda r:hashlib.sha256(f'{SEED}:{r["qid"]}'.encode()).hexdigest());records=[];used=set()
def emit(r,target,claimed,cause,partition):
 im=onfile[r['image_id']];dest=root/'images'/Path(im['image']).name
 if not dest.exists():os.link(im['image'],dest)
 q=phrase(r['kind'],r['verb'],r['x'],claimed,rng)
 records.append(dict(id=f'bool_abstain:{r["qid"]}',source='bool_abstain',source_split='tdiuc-'+r['qid'].split('-')[0],source_group=str(r['image_id']),family='false_premise_claim' if cause else r['kind']+'_claim',
  license='CC-BY-4.0 (TDIUC annotation JSON); images: COCO/Flickr terms, not relicensed',images=[dict(im,image=str(dest))],request=dict(request_id=f'bool-abstain-{r["qid"]}',state={},fields=[dict(id='answer',type='boolean',question=q)]),
  target=target,abstention_cause=cause,source_answer=r['answer'],partition=partition))
quota=dict(fit=dict(unknown=N_UNKNOWN,true=N_UNKNOWN//2,false=N_UNKNOWN//2),test=dict(unknown=N_TEST//2,true=N_TEST//4,false=N_TEST//4))
for r in rows:
 s=side[r['image_id']]
 if len(s)!=1 or r['image_id'] in used:continue
 where=next(iter(s))
 if where=='fit' and (r['image_id'] in banned_coco or onfile[r['image_id']]['sha256'] in banned_sha):continue
 q=quota[where];partition='test' if where=='test' else ('dev' if rng.random()<.03 else 'train')
 if r['family']=='absurd':
  if q['unknown']<=0:continue
  emit(r,None,rng.choice(values[r['kind']]),'false_premise',partition);q['unknown']-=1
 else:
  if r['answer'] not in values[r['kind']]:continue
  want='true' if q['true']>=q['false'] else 'false'
  if q[want]<=0:continue
  if want=='true':emit(r,True,r['answer'],None,partition)
  else:
   wrong=[v for v in values[r['kind']] if not clash(v,r['answer'])]
   if not wrong:continue
   emit(r,False,rng.choice(wrong),None,partition)
  q[want]-=1
 used.add(r['image_id'])
(root/'records.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
(root/'README.md').write_text(f'''# bool_abstain\n\nYes/no claims built from TDIUC (COCO images already fetched by vqav2/tdiuc/aokvqa; hardlinked here).\n\n- false-premise claims (target unknown): the claim is about an object TDIUC marks absent (its "absurd" questions, derived from COCO annotations).\n- answerable claims from TDIUC color/attribute questions, true and false in equal numbers, same templates, so wording does not reveal the answer. False claims avoid confusable values (gray/silver, brown/tan, ...).\n- One claim per image; images on the evaluation exclusion list never enter train/dev; test claims use only images that sit in other sources' test partitions.\n\nCounts: {dict(Counter((r["partition"],str(r["target"])) for r in records))}\n\nLicence: annotations CC BY 4.0 as declared in the TDIUC JSON (partially verified, see data/decision-v1/tdiuc/README.md); COCO images carry Flickr terms and are not relicensed.\nLimit: absence is as reliable as TDIUC's absurd-question generation; phrasing is templated (5 per kind).\n''')
print(len(records),dict(Counter((r['partition'],str(r['target'])) for r in records)));print(*[r['request']['fields'][0]['question']+' -> '+str(r['target']) for r in records[:6]],sep='\n')
