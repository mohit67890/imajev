"""Acquire pinned public claims/reading datasets and record license evidence.

This script never interprets a model/dataset-card tag as a license. Sources without an
explicit primary dataset license are downloaded for audit but marked quarantined.
Downloads are resumable through HTTP Range requests and are checksum inventoried.
"""
import argparse, csv, hashlib, json, urllib.request, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/decision-v1-text/raw'; LICENSES=ROOT/'data/decision-v1-text/licenses'; REPORT=ROOT/'reports/v1.1-datasets/claims-reading.json'
CC_BY_SA_3='https://creativecommons.org/licenses/by-sa/3.0/legalcode'

SOURCES={
 'fever':dict(revision='FEVER v1.0 / Zenodo 4925954',status='admitted',spdx='CC-BY-SA-3.0',license='CC BY-SA 3.0 / applicable Wikipedia article terms',license_urls=['https://fever.ai/download/fever/license.html','https://fever.ai/dataset/fever.html'],notes='165,447 unique claims: 145,449 train plus 19,998 shared-task dev. paper_dev and paper_test partition shared_task_dev and are not additional unique claims. wiki-pages.zip supplies the referenced evidence sentences.',unique_rows=165447,files={
  'train.jsonl':'https://fever.ai/download/fever/train.jsonl','shared_task_dev.jsonl':'https://fever.ai/download/fever/shared_task_dev.jsonl','paper_dev.jsonl':'https://fever.ai/download/fever/paper_dev.jsonl','paper_test.jsonl':'https://fever.ai/download/fever/paper_test.jsonl','wiki-pages.zip':'https://fever.ai/download/fever/wiki-pages.zip'}),
 'boolq':dict(revision='35b264d03638db9f4ce671b711558bf7ff0f80d5',status='admitted',spdx='CC-BY-SA-3.0',license='CC BY-SA 3.0',license_urls=['https://raw.githubusercontent.com/google-research-datasets/boolean-questions/90af34107399cc7a446b373dc4ee35b8001da7c2/README.md',CC_BY_SA_3],files={
  'train.parquet':'https://huggingface.co/datasets/google/boolq/resolve/35b264d03638db9f4ce671b711558bf7ff0f80d5/data/train-00000-of-00001.parquet','validation.parquet':'https://huggingface.co/datasets/google/boolq/resolve/35b264d03638db9f4ce671b711558bf7ff0f80d5/data/validation-00000-of-00001.parquet'}),
 'wanli':dict(revision='61c95318fd71c55b6ba355d76253254615f387ec',status='quarantined',license='NO DATASET LICENSE FOUND IN PRIMARY REPOSITORY',license_urls=['https://huggingface.co/datasets/alisawuffles/WANLI/raw/61c95318fd71c55b6ba355d76253254615f387ec/README.md'],files={
  x:f'https://huggingface.co/datasets/alisawuffles/WANLI/resolve/61c95318fd71c55b6ba355d76253254615f387ec/{x}' for x in ('train.jsonl','test.jsonl','anonymized_annotations.jsonl')}),
 'snli':dict(revision='cdb5c3d5eed6ead6e5a341c8e56e669bb666725b',status='admitted',spdx='CC-BY-SA-4.0',license='CC BY-SA 4.0',license_urls=['https://nlp.stanford.edu/projects/snli/','https://creativecommons.org/licenses/by-sa/4.0/legalcode'],files={
  f'{s}.parquet':f'https://huggingface.co/datasets/stanfordnlp/snli/resolve/cdb5c3d5eed6ead6e5a341c8e56e669bb666725b/plain_text/{s}-00000-of-00001.parquet' for s in ('train','validation','test')}),
 'quality':dict(revision='f84977c40dbfef70c9cab48037b7becfc8e45f73 (contains QuALITY v1.0.1)',status='quarantined',license='NO DATASET LICENSE FOUND IN PRIMARY REPOSITORY',license_urls=['https://raw.githubusercontent.com/nyu-mll/quality/f84977c40dbfef70c9cab48037b7becfc8e45f73/README.md'],files={
  f'QuALITY.v1.0.1.htmlstripped.{s}':f'https://raw.githubusercontent.com/nyu-mll/quality/f84977c40dbfef70c9cab48037b7becfc8e45f73/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.{s}' for s in ('train','dev','test')}),
 'cosmosqa':dict(revision='b6eb99cca4e2a51dd28a9a6f562534872d851639',status='quarantined',license='NO DATASET LICENSE FOUND IN PRIMARY REPOSITORY',license_urls=['https://raw.githubusercontent.com/wilburOne/cosmosqa/b6eb99cca4e2a51dd28a9a6f562534872d851639/README.md'],files={
  'train.csv':'https://raw.githubusercontent.com/wilburOne/cosmosqa/b6eb99cca4e2a51dd28a9a6f562534872d851639/data/train.csv','valid.csv':'https://raw.githubusercontent.com/wilburOne/cosmosqa/b6eb99cca4e2a51dd28a9a6f562534872d851639/data/valid.csv','test.jsonl':'https://raw.githubusercontent.com/wilburOne/cosmosqa/b6eb99cca4e2a51dd28a9a6f562534872d851639/data/test.jsonl'}),
 'socialiqa':dict(revision='public socialiqa-train-dev.zip',status='quarantined',license='NO DATASET LICENSE INCLUDED BY AUTHORITATIVE DOWNLOAD',license_urls=['https://huggingface.co/datasets/allenai/social_i_qa/raw/8835ceb9141d7896d9d968634a9b21ae440e3ec5/README.md'],files={'socialiqa-train-dev.zip':'https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip'}),
 'commonsenseqa':dict(revision='94630fe30dad47192a8546eb75f094926d47e155',status='quarantined',license='NO DATASET LICENSE FOUND IN PRIMARY REPOSITORY',license_urls=['https://raw.githubusercontent.com/jonathanherzig/commonsenseqa/master/README.md'],files={
  f'{s}.parquet':f'https://huggingface.co/datasets/tau/commonsense_qa/resolve/94630fe30dad47192a8546eb75f094926d47e155/data/{s}-00000-of-00001.parquet' for s in ('train','validation','test')})}

def download(url,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():return
 part=dst.with_suffix(dst.suffix+'.part');start=part.stat().st_size if part.exists() else 0
 req=urllib.request.Request(url,headers={'User-Agent':'imajev-dataset-audit/1.1',**({'Range':f'bytes={start}-'} if start else {})})
 with urllib.request.urlopen(req) as r:
  if start and r.status!=206:start=0
  with part.open('ab' if start else 'wb') as f:
   while chunk:=r.read(1<<20):f.write(chunk)
 part.replace(dst)

def count(path):
 if path.suffix in ('.jsonl','') or '.htmlstripped.' in path.name:return sum(1 for x in path.open(errors='replace') if x.strip())
 if path.suffix=='.csv':
  with path.open(errors='replace',newline='') as f:return max(0,sum(1 for _ in csv.reader(f))-1)
 if path.suffix=='.parquet':
  import pyarrow.parquet as pq
  return pq.ParquetFile(path).metadata.num_rows
 if path.suffix=='.zip':
  with zipfile.ZipFile(path) as z:
   splits={x.filename:sum(1 for line in z.open(x) if line.strip()) for x in z.infolist()
           if x.filename.endswith('.jsonl') and not x.filename.startswith('__MACOSX/')}
   return {'rows':sum(splits.values()),'splits':splits,'members':len(z.infolist()),'uncompressed_bytes':sum(x.file_size for x in z.infolist())}

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',action='append',choices=sorted(SOURCES));a=p.parse_args();inventory=[]
 for name in a.source or SOURCES:
  spec=SOURCES[name];entry={k:spec[k] for k in ('revision','status','license')};entry['source']=name;entry['files']=[];entry['license_evidence']=[];entry['notes']=spec.get('notes');entry['unique_rows']=spec.get('unique_rows')
  for i,url in enumerate(spec['license_urls']):
   first='LICENSE.txt' if spec['status']=='admitted' else 'NO-LICENSE-EVIDENCE.txt'
   dst=LICENSES/name/(first if i==0 else f'SOURCE-{i+1}.txt');download(url,dst)
   entry['license_evidence'].append({'path':str(dst.relative_to(ROOT)),'url':url,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
  if spec['status']=='admitted':
   evidence=LICENSES/name/'LICENSE.txt';digest=hashlib.sha256(evidence.read_bytes()).hexdigest()
   receipt={'spdx':spec['spdx'],'evidence_sha256':digest,'commercial_use_reviewed':True,'source':name,'source_url':spec['license_urls'][0],'reviewed_at':'2026-09-22','reviewed_by':'Codex source-file review','review_note':spec['license']}
   (LICENSES/name/'LICENSE.txt.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
   (LICENSES/name/'review.json').write_text(json.dumps({'license_status':'verified_primary_data_license_sharealike','spdx':spec['spdx'],'training_admitted':True,'evidence':str(evidence.relative_to(ROOT)),'source_url':spec['license_urls'][0],'notes':spec['license'],'evidence_sha256':digest},indent=2)+'\n')
  for filename,url in spec['files'].items():
   dst=RAW/name/filename;download(url,dst);entry['files'].append({'path':str(dst.relative_to(ROOT)),'url':url,'bytes':dst.stat().st_size,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'rows':count(dst)})
  entry['training_eligible']=spec['status']=='admitted';entry['blocker']=None if entry['training_eligible'] else spec['license'];inventory.append(entry)
 REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(json.dumps({'schema_version':1,'sources':inventory},indent=2)+'\n');print(REPORT)
if __name__=='__main__':main()
