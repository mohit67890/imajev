"""Convert acquired routing and GoEmotions sources into licensed decision records."""
from __future__ import annotations
import csv,hashlib,json
from collections import Counter
from pathlib import Path
from v1_text.common import fit_partition,verified_license,write_jsonl

ROOT=Path('data/decision-v1-text');OUT=ROOT/'converted/routing-emotion.jsonl';REPORT=Path('reports/v1.1-datasets/routing-emotion-conversion.json')
PHRASES=("Which intent best matches this request?","Route this request to the correct intent.","Classify the user's request.","Choose the most appropriate request category.","What kind of request is this?","Select the workflow that should handle this request.")
EMOTION_PHRASES=("The text expresses {label}.","Is {label} present in this text?","Does the writer show {label}?","Classify whether the text conveys {label}.","The writer's emotion includes {label}.","Mark whether {label} applies.")
CAPS={'banking77':12000,'clinc150':12000,'massive':17000,'snips':12000,'goemotions':12000};EVAL_CAP=1000

def rank(*parts):return hashlib.sha256('\0'.join(map(str,parts)).encode()).hexdigest()
def group(source,text):return f"{source}:{hashlib.sha256(' '.join(text.casefold().split()).encode()).hexdigest()}"
def request(source,rid,text,field):return {'schema_version':'1.0','request_id':f'{source}-{rid}','state':text,'fields':[field]}
def choice_record(source,split,text,label,labels,license_info,uid,locale=None,oos=False):
 labels=sorted(labels,key=lambda x:rank(source,uid,'option',x));phrase=PHRASES[int(rank(source,uid,'prompt')[:8],16)%len(PHRASES)]
 field={'id':'intent','type':'choice','question':phrase,'options':[{'value':x} for x in labels]}
 target=None if oos else label;part='test' if split=='test' else 'dev' if split in {'dev','validation','val'} else fit_partition(group(source,text))
 return {'id':f'{source}:{uid}','source':source,'source_split':split,'source_group':group(source,text),'family':'intent_routing','license':license_info,'images':[],
  'request':request(source,uid,text,field),'target':target,'abstention_cause':'not_listed' if oos else None,'partition':part,'source_answer':label,
  **({'locale':locale} if locale else {})}
def license_for(registry,name):
 entry=next(x for x in registry['sources'] if x['source']==name)
 if not entry.get('training_admitted'):raise ValueError(f'{name} is not admitted by source registry')
 v=entry.get('verified_license_evidence') or []; 
 if not v:raise ValueError(f'{name} lacks verified license evidence')
 got=verified_license(Path(v[0]['evidence']),v[0]['spdx']);return got
def load_candidates(registry):
 out=[]
 # Banking77
 lic=license_for(registry,'banking77');labels=json.loads((ROOT/'raw/banking77/categories.json').read_text())
 for split in ('train','test'):
  with (ROOT/f'raw/banking77/{split}.csv').open(newline='') as f:
   for n,r in enumerate(csv.DictReader(f)):out.append(choice_record('banking77',split,r['text'],r['category'],labels,lic,f'{split}-{n}'))
 # CLINC150; OOS is a real unknown, never an ordinary option.
 lic=license_for(registry,'clinc150');data=json.loads((ROOT/'raw/clinc150/data_full.json').read_text());labels=sorted({label for k,v in data.items() if not k.startswith('oos_') for _,label in v})
 for key,values in data.items():
  split={'val':'validation','oos_val':'validation','test':'test','oos_test':'test'}.get(key,'train')
  for n,(text,label) in enumerate(values):out.append(choice_record('clinc150',split,text,label,labels,lic,f'{key}-{n}',oos=key.startswith('oos_')))
 # MASSIVE six requested locales.
 lic=license_for(registry,'massive');paths=sorted((ROOT/'raw/massive/selected').glob('*.jsonl'));allrows=[json.loads(x) for p in paths for x in p.read_text().splitlines() if x.strip()];labels=sorted({r['intent'] for r in allrows})
 for r in allrows:out.append(choice_record('massive',r['partition'],r['utt'],r['intent'],labels,lic,f"{r['locale']}-{r['partition']}-{r['id']}",r['locale']))
 # SNIPS official full train + validation.
 lic=license_for(registry,'snips');labels=sorted(p.name for p in (ROOT/'raw/snips').iterdir() if p.is_dir() and p.name!='2017-06-custom-intent-engines')
 for label in labels:
  for p in sorted((ROOT/'raw/snips'/label).glob('*.json')):
   if p.name.startswith('train_') and not p.name.endswith('_full.json'):continue
   split='validation' if p.name.startswith('validate_') else 'train';items=json.loads(p.read_text(encoding='latin-1'))[label]
   for n,item in enumerate(items):
    text=''.join(x['text'] for x in item['data']);out.append(choice_record('snips',split,text,label,labels,lic,f'{split}-{label}-{n}'))
 # GoEmotions: one noul decision per (text, emotion); cap below keeps a deterministic useful subset.
 lic=license_for(registry,'goemotions');emotions=(ROOT/'raw/goemotions/emotions.txt').read_text().splitlines()
 for split in ('train','dev','test'):
  with (ROOT/f'raw/goemotions/{split}.tsv').open() as f:
   for n,line in enumerate(f):
    text,label_ids,example_id=line.rstrip('\n').split('\t');positive={int(x) for x in label_ids.split(',')}
    for i,label in enumerate(emotions):
     uid=f'{split}-{example_id}-{i}';q=EMOTION_PHRASES[int(rank('goemotions',uid,'prompt')[:8],16)%len(EMOTION_PHRASES)].format(label=label.replace('_',' '))
     field={'id':'applies','type':'boolean','question':q,'yes_description':f'{label} is expressed','no_description':f'{label} is not expressed'}
     part='test' if split=='test' else 'dev' if split=='dev' else fit_partition(group('goemotions',text))
     out.append({'id':f'goemotions:{uid}','source':'goemotions','source_split':split,'source_group':group('goemotions',text),'family':f'emotion_{label}','license':lic,'images':[],
      'request':request('goemotions',uid,text,field),'target':i in positive,'abstention_cause':None,'partition':part,'source_answer':i in positive})
 return out
def select(rows):
 # Identical normalized text is one group. Official test wins, then dev; lower-priority copies are removed.
 priority={'test':3,'dev':2,'calibration':1,'train':0};best={}
 for r in rows:best[r['source_group']]=max(best.get(r['source_group'],'train'),r['partition'],key=priority.get)
 rows=[r for r in rows if r['partition']==best[r['source_group']]];selected=[]
 for source in CAPS:
  mine=[r for r in rows if r['source']==source]
  for part in ('test','dev','calibration','train'):
   pool=sorted((r for r in mine if r['partition']==part),key=lambda r:rank('select',r['id']))
   cap=EVAL_CAP if part in {'test','dev'} else CAPS[source]
   if source=='goemotions' and part=='train' and pool:
    # Uniform sampling over 28 one-vs-rest probes makes an all-false classifier
    # roughly 96% accurate. Balance the training decisions while leaving
    # calibration and official evaluation at their natural prevalence.
    halves=[]
    for target in (True,False):
     by_family={family:[] for family in sorted({r['family'] for r in pool})}
     for row in pool:
      if row['target'] is target:by_family[row['family']].append(row)
     wanted=cap//2;take=[];offset=0
     while len(take)<wanted:
      added=False
      for family in sorted(by_family):
       values=by_family[family]
       if offset<len(values):take.append(values[offset]);added=True
       if len(take)==wanted:break
      if not added:break
      offset+=1
     halves.extend(take)
    if len(halves)!=cap:raise ValueError(f'cannot balance GoEmotions train at cap {cap}: got {len(halves)}')
    selected.extend(halves)
   else:selected.extend(pool[:cap])
 return sorted(selected,key=lambda r:r['id'])
def main():
 registry=json.loads((ROOT/'source-registry.json').read_text());allrows=load_candidates(registry);rows=select(allrows);write_jsonl(OUT,rows)
 partitions={}
 for r in rows:partitions.setdefault(r['source_group'],set()).add(r['partition'])
 report={'output':str(OUT),'records':len(rows),'candidate_records':len(allrows),'by_source_partition':{f'{s}/{p}':n for (s,p),n in sorted(Counter((r['source'],r['partition']) for r in rows).items())},
  'by_source':dict(Counter(r['source'] for r in rows)),'unknown':sum(r['target'] is None for r in rows),'boolean_true':sum(r['target'] is True for r in rows),
  'goemotions_boolean_by_partition':{p:{str(v).lower():n for v,n in Counter(r['target'] for r in rows if r['source']=='goemotions' and r['partition']==p).items()} for p in ('train','calibration','dev','test')},
  'prompt_phrasings':6,'eval_cap_per_source_partition':EVAL_CAP,'training_cap_per_source':CAPS,
  'group_leaks':sum(len(parts)>1 for parts in partitions.values())}
 REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
