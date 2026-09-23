"""Convert licensed ESCI judgments and ABO metadata to grounded JSON decisions."""
from __future__ import annotations
import argparse, collections, gzip, hashlib, json, random, re, tarfile
from pathlib import Path
import pyarrow.parquet as pq
from .common import fit_partition, verified_license, write_jsonl

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/decision-v1-text'
PROMPTS=['According to the listing, what is its {key}?','Read the listing and select its {key}.','Which {key} is explicitly recorded in this listing?','Extract the listed {key}.','Identify the {key} from the supplied metadata.','What value does the metadata give for {key}?']
RELEVANCE=['How relevant is this product to the query?','Rate the product’s relevance to the search query.','Assess the match between query and listing.','Score how well this listing satisfies the query.','Using the rubric, judge this query–product pair.','Assign a relevance level to the supplied search result.']
LEVELS=[{'value':0,'description':'Irrelevant: does not satisfy the query.'},{'value':1,'description':'Complement: useful with the requested product, but not itself a substitute.'},{'value':2,'description':'Substitute: serves the same purpose with differences.'},{'value':3,'description':'Exact: satisfies the query requirements.'}]
def keyhash(x):return hashlib.sha256(str(x).encode()).hexdigest()
def license_for(source):
 registry=json.loads((DATA/'source-registry.json').read_text())
 entry=next(s for s in registry['sources'] if s['source']==source)
 if not entry.get('training_admitted'):raise ValueError(f'{source} is not admitted')
 lic=entry['verified_license_evidence'][0]
 return verified_license(Path(lic['evidence']),lic['spdx'])
def record(source,ident,group,split,partition,state,field,target,lic):
 return dict(id=f'{source}:text:{ident}',source=source,source_group=group,source_split=split,partition=partition,family=f'{source}_structured',license=lic,images=[],request={'schema_version':'1.0','request_id':f'{source}-{ident}','state':state,'fields':[field]},target=target,abstention_cause=None)
def abo_image_groups(manifest):
 """Return unambiguous item -> (group, partition) links from admitted image records."""
 links=collections.defaultdict(set);group_partitions=collections.defaultdict(set)
 with Path(manifest).open() as f:
  for line in f:
   r=json.loads(line)
   if r['source']!='abo':continue
   group_partitions[r['source_group']].add(r['partition'])
   for item in re.findall(r'B[0-9A-Z]{9}',r['id']):links[item].add((r['source_group'],r['partition']))
 resolved={}
 for item,values in links.items():
  if len(values)!=1:continue
  group,partition=next(iter(values))
  if len(group_partitions[group])==1:resolved[item]=(group,partition)
 return resolved,set(links)-set(resolved)
def esci(limit=12000):
 base=DATA/'raw/esci/shopping_queries_dataset';selected=[];counts=collections.Counter();lic=license_for('esci')
 # Query text, not row id, defines the split group. Test occurrences win.
 test_queries=set()
 examples=base/'shopping_queries_dataset_examples.parquet'
 for b in pq.ParquetFile(examples).iter_batches(batch_size=16384,columns=['query','split']):
  test_queries.update(keyhash(str(r['query']).strip().casefold()) for r in b.to_pylist() if r['split']=='test')
 for b in pq.ParquetFile(examples).iter_batches(batch_size=16384):
  for r in b.to_pylist():
   group=keyhash(str(r['query']).strip().casefold());partition='test' if r['split']=='test' else fit_partition('esci:'+group)
   if r['split']!='test' and group in test_queries:continue
   cap=limit if partition=='train' else 600
   if counts[partition]>=cap:continue
   selected.append((r,group,partition));counts[partition]+=1
 needed={(r['product_id'],r['product_locale']) for r,_,_ in selected};products={}
 for b in pq.ParquetFile(base/'shopping_queries_dataset_products.parquet').iter_batches(batch_size=4096):
  for p in b.to_pylist():
   key=(p['product_id'],p['product_locale'])
   if key in needed:products[key]=p
 out=[]
 for r,group,partition in selected:
  p=products.get((r['product_id'],r['product_locale']))
  if p is None:raise ValueError('Missing ESCI product join')
  state={'query':r['query'],'listing':{k:p[k] for k in ['product_title','product_description','product_bullet_point','product_brand','product_color'] if p.get(k)}}
  if len(json.dumps(state,ensure_ascii=False))>9500:continue
  field={'id':'relevance','type':'ordinal','question':RELEVANCE[int(group[:8],16)%6],'levels':LEVELS}
  out.append(record('esci',r['example_id'],group,r['split'],partition,state,field,{'I':0,'C':1,'S':2,'E':3}[r['esci_label']],lic))
 return out

def abo(limit=8000):
 lic=license_for('abo');image_groups,ambiguous_items=abo_image_groups(ROOT/'data/manifests/decision-v1.jsonl')
 pool=[];seen=set()
 with tarfile.open(DATA/'raw/abo/abo-listings.tar') as archive:
  for member in archive:
   if not member.name.endswith('.json.gz'):continue
   with gzip.GzipFile(fileobj=archive.extractfile(member)) as f:
    for line in f:
     r=json.loads(line);ident=r['item_id']
     if ident in seen:continue
     seen.add(ident);state={}
     if ident in ambiguous_items:continue
     for key in ('item_name','brand','color','product_type','model_number'):
      vals=r.get(key,[]);english=[v['value'] for v in vals if v.get('language_tag','en_US').startswith('en') and v.get('value')]
      if english:state[key]=english[0]
     if 'item_name' not in state:continue
     group,partition=image_groups.get(ident,('item|'+ident,fit_partition('abo:'+ident)))
     pool.append((ident,group,partition,state))
 vocab={k:sorted({str(s[k]) for _,_,_,s in pool if k in s}) for k in ('brand','color','product_type')}
 out=[];counts=collections.Counter()
 for ident,group,partition,state in sorted(pool,key=lambda r:keyhash(r[0])):
  cap=limit if partition=='train' else 500
  if counts[partition]>=cap:continue
  available=[k for k in vocab if k in state and len(vocab[k])>3]
  if not available:continue
  rng=random.Random(ident);key=rng.choice(available);answer=str(state[key]);options=rng.sample([v for v in vocab[key] if v!=answer],3)+[answer];rng.shuffle(options)
  field={'id':'metadata','type':'choice','question':PROMPTS[int(keyhash(ident)[:8],16)%6].format(key=key.replace('_',' ')),'options':[{'value':v} for v in options]}
  out.append(record('abo',ident,group,'all',partition,{'listing':state},field,answer,lic));counts[partition]+=1
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DATA/'converted/structured.jsonl');a=p.parse_args()
 rows=esci()+abo();write_jsonl(a.output,rows)
 report={'records':len(rows),'counts':dict(collections.Counter(f"{r['source']}/{r['partition']}" for r in rows)),'notes':['ESCI query text groups cannot cross partitions; official test stays test.','ABO item partitions align with existing image records where available.','Long ESCI listings over 9500 characters omitted, not silently truncated.']}
 path=ROOT/'reports/v1.1-datasets/structured-conversion.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
