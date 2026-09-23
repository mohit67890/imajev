"""Acquire public reading/structured-text sources, pinned metadata and byte hashes.

Data acquisition does not imply licence approval. Licences are reviewed separately.
"""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import shutil
import time
from pathlib import Path
import requests
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/decision-v1-text'
REPORT=ROOT/'reports/v1.1-datasets/reading-structured.json'
SOURCES={
 'arc':('allenai/ai2_arc',('ARC-',)),
 'openbookqa':('allenai/openbookqa',('main/','additional/')),
 'hellaswag':('Rowan/hellaswag',('data/',)),
 'copa':('aps/super_glue',('copa/',)),
 'squad2':('rajpurkar/squad_v2',('squad_v2/',)),
}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def download(url,path,size=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 receipt=path.with_name(path.name+'.download.json')
 if path.is_file() and receipt.exists():
  old=json.loads(receipt.read_text())
  if old.get('url')==url and (size is None or path.stat().st_size==size) and digest(path)==old['sha256']:return old
 for attempt in range(3):
  partial=path.with_name(path.name+'.partial')
  try:
   with requests.get(url,stream=True,timeout=(20,90)) as r:
    r.raise_for_status()
    with partial.open('wb') as f:
     for chunk in r.iter_content(1024*1024):f.write(chunk)
   if size is not None and partial.stat().st_size!=size:raise ValueError(f'Size mismatch for {url}')
   partial.replace(path)
   entry={'url':url,'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':digest(path)}
   if path.suffix=='.parquet':entry['rows']=pq.ParquetFile(path).metadata.num_rows
   receipt.write_text(json.dumps(entry,indent=2)+'\n');return entry
  except Exception:
   if attempt==2:raise
   time.sleep(attempt+1)

def acquire_hf(name,repo,prefixes):
 out=DATA/'raw'/name;out.mkdir(parents=True,exist_ok=True)
 meta_file=out/'source.json'
 if meta_file.exists():meta=json.loads(meta_file.read_text())
 else:
  r=requests.get(f'https://huggingface.co/api/datasets/{repo}',timeout=40);r.raise_for_status();meta=r.json();meta_file.write_text(json.dumps(meta,indent=2)+'\n')
 revision=meta['sha']
 r=requests.get(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&limit=1000',timeout=40);r.raise_for_status()
 tree=r.json();(out/'tree.json').write_text(json.dumps(tree,indent=2)+'\n')
 selected=[x for x in tree if x['type']=='file' and (x['path']=='README.md' or any(x['path'].startswith(p) for p in prefixes))]
 def one(x):return download(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{x["path"]}',out/x['path'],x['size'])
 with concurrent.futures.ThreadPoolExecutor(4) as pool:files=list(pool.map(one,selected))
 return {'source':name,'repository':repo,'revision':revision,'acquisition_status':'downloaded','license_status':'pending_primary_file_review','training_admitted':False,'files':files,'rows':sum(x.get('rows',0) for x in files)}

def acquire_esci():
 name='esci';out=DATA/'raw'/name;out.mkdir(parents=True,exist_ok=True)
 meta_file=out/'tree.json'
 if meta_file.exists():tree=json.loads(meta_file.read_text())
 else:
  r=requests.get('https://api.github.com/repos/amazon-science/esci-data/git/trees/main?recursive=1',timeout=40);r.raise_for_status();tree=r.json();meta_file.write_text(json.dumps(tree,indent=2)+'\n')
 rev=tree['sha'];files=[]
 for item in tree['tree']:
  p=item['path']
  if p in ('LICENSE','README.md','NOTICE') or p.startswith('shopping_queries_dataset/') and p.endswith('.parquet'):
   host='media.githubusercontent.com/media' if p.endswith('.parquet') else 'raw.githubusercontent.com'
   files.append(download(f'https://{host}/amazon-science/esci-data/{rev}/{p}',out/p))
 return {'source':name,'repository':'amazon-science/esci-data','revision':rev,'acquisition_status':'downloaded','license_status':'pending_primary_file_review','training_admitted':False,'files':files,'rows':sum(x.get('rows',0) for x in files)}

def reuse_abo():
 source=ROOT/'.cache/datasets/v1/abo';out=DATA/'raw/abo';out.mkdir(parents=True,exist_ok=True)
 files=[]
 for name in ('abo-listings.tar','README.md','listings-README.md','LICENSE-CC-BY-4.0.txt'):
  src=source/name;dest=out/name
  if not dest.exists():shutil.copyfile(src,dest)
  files.append({'path':str(dest.relative_to(ROOT)),'reused_from':str(src.relative_to(ROOT)),'bytes':dest.stat().st_size,'sha256':digest(dest)})
 return {'source':'abo','acquisition_status':'cached_complete_metadata','license_status':'pending_primary_file_review','training_admitted':False,'files':files}

def main():
 REPORT.parent.mkdir(parents=True,exist_ok=True)
 results=[]
 jobs=[(name,lambda n=name,r=repo,p=prefixes:acquire_hf(n,r,p)) for name,(repo,prefixes) in SOURCES.items()]+[('esci',acquire_esci),('abo',reuse_abo)]
 # Bound concurrent downloads to keep the shared connection usable by the other source groups.
 with concurrent.futures.ThreadPoolExecutor(3) as pool:
  futures={pool.submit(fn):name for name,fn in jobs}
  for fut in concurrent.futures.as_completed(futures):
   name=futures[fut]
   try:entry=fut.result()
   except Exception as exc:entry={'source':name,'acquisition_status':'blocked','error':str(exc),'training_admitted':False}
   review=DATA/'licenses'/name/'review.json'
   if review.exists():entry.update(json.loads(review.read_text()))
   results.append(entry);REPORT.write_text(json.dumps({'sources':sorted(results,key=lambda x:x['source'])},indent=2)+'\n')
   print(json.dumps({'source':name,'status':entry['acquisition_status'],'rows':entry.get('rows'),'error':entry.get('error')}),flush=True)

if __name__=='__main__':main()
