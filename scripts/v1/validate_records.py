"""Gate for every decision-v1 converter: run `python scripts/v1/validate_records.py <source>` and fix until it passes."""
import json,sys,hashlib
from pathlib import Path
from collections import Counter
from PIL import Image
sys.path.insert(0,'src')
from vision_decision.contracts import Request

CAUSES={None,'not_listed','false_premise','insufficient_evidence','mismatched_reference'};PARTITIONS={'train','dev','test'}
source=sys.argv[1];root=Path('data/decision-v1')/source;rows=[json.loads(l) for l in (root/'records.jsonl').read_text().splitlines()]
excl=json.loads(Path('data/decision-v1/exclusions.json').read_text());banned=set(excl['sha256']);errors=[];ids=set();groups={}
for n,r in enumerate(rows):
 def bad(msg):errors.append(f'{r.get("id",n)}: {msg}')
 for k in ('id','source','source_split','source_group','family','license','images','request','target','abstention_cause','partition','source_answer'):
  if k not in r:bad(f'missing {k}')
 if errors and errors[-1].startswith(str(r.get('id',n))+': missing'):continue
 if r['id'] in ids:bad('duplicate id')
 ids.add(r['id'])
 if r['partition'] not in PARTITIONS:bad('bad partition')
 if r['abstention_cause'] not in CAUSES:bad('bad abstention_cause')
 if (r['target'] is None)!=(r['abstention_cause'] is not None):bad('target None <=> abstention_cause set')
 groups.setdefault(r['source_group'],set()).add('test' if r['partition']=='test' else 'fit')
 try:
  req=Request.model_validate(r['request']);assert len(req.fields)==1;f=req.fields[0]
  if r['target'] is not None:
   if f.type=='boolean':assert isinstance(r['target'],bool),'boolean target must be true/false'
   elif f.type=='choice':assert r['target'] in [o.value for o in f.options],'target not among options'
   else:assert r['target'] in [l.value for l in f.levels],'target not among levels'
  if f.type=='choice' and r['abstention_cause']=='not_listed':assert r['source_answer'] not in [o.value for o in f.options],'not_listed but answer present'
 except Exception as e:bad(f'request/target: {e}')
 if not 1<=len(r['images'])<=2:bad('need 1-2 images')
 for im in r['images']:
  p=Path(im['image'])
  if not p.is_file():bad(f'missing image {p}');continue
  if im['sha256'] in banned and r['partition']!='test':bad('image was used by an earlier evaluation')
  if n%200==0:  # spot-check bytes and decodability
   if hashlib.sha256(p.read_bytes()).hexdigest()!=im['sha256']:bad('sha256 mismatch')
   with Image.open(p) as x:
    x.load()
    if x.width*x.height>1_050_000:bad('image larger than the 1 MP storage cap')
leak=[g for g,s in groups.items() if len(s)>1]
if leak:errors.append(f'{len(leak)} source_groups appear in both test and train/dev, e.g. {leak[:3]}')
summary=dict(source=source,records=len(rows),partitions=dict(Counter(r['partition'] for r in rows)),families=dict(Counter(r['family'] for r in rows)),
 field_types=dict(Counter(r['request']['fields'][0]['type'] for r in rows)),abstention=dict(Counter(str(r['abstention_cause']) for r in rows)),
 option_counts=dict(sorted(Counter(len(r['request']['fields'][0].get('options',r['request']['fields'][0].get('levels',[]))) for r in rows).items())),two_image=sum(len(r['images'])==2 for r in rows),errors=len(errors))
(root/'validation.json').write_text(json.dumps(dict(summary=summary,first_errors=errors[:50]),indent=2)+'\n');print(json.dumps(summary,indent=2))
if errors:print('\n'.join(errors[:20]));sys.exit(1)
print('VALID')
