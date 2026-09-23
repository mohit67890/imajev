"""Consolidate acquisition-group inventories and verify on-disk file hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
try:
 from .common import verified_license
except ImportError:
 from common import verified_license

ROOT=Path(__file__).resolve().parents[2]
EXPECTED=('typed_decisions banking77 clinc150 massive snips spamassassin enron_spam civil_comments aegis openai_moderation goemotions sst5 helpsteer2 ultrafeedback onestopenglish fever boolq wanli snli quality cosmosqa socialiqa commonsenseqa openbookqa arc hellaswag copa squad2 esci abo mmlu').split()
ALIASES={'typed-decisions':'typed_decisions','enron-spam':'enron_spam','civil-comments':'civil_comments','openai-moderation':'openai_moderation','one_stop_english':'onestopenglish','onestop_english':'onestopenglish','squad_v2':'squad2','squad_2':'squad2','aegis_2':'aegis','aegis2':'aegis','nvidia_aegis':'aegis','openai_moderation_dataset':'openai_moderation','sst-5':'sst5'}

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def consolidate(paths,verify=False):
 sources={}
 for path in paths:
  report=json.loads(path.read_text())
  for original in report.get('sources',[]):
   entry=dict(original);name=ALIASES.get(entry['source'],entry['source']);entry['source']=name
   entry['inventory_file']=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
   if name not in EXPECTED:raise ValueError(f'Unexpected source: {name}')
   if name in sources:raise ValueError(f'Duplicate source inventory: {name}')
   sources[name]=entry
 for name in EXPECTED:
  sources.setdefault(name,{'source':name,'acquisition_status':'not_acquired','training_admitted':False,'files':[]})
 for entry in sources.values():
  files=entry.get('files',[]);valid=[];errors=[]
  for item in files:
   if not item.get('path'):continue
   path=Path(item['path']);path=path if path.is_absolute() else ROOT/path
   if not path.is_file():errors.append(f'Missing {item["path"]}');continue
   if 'bytes' in item and path.stat().st_size!=item['bytes']:errors.append(f'Size mismatch {item["path"]}');continue
   if verify and item.get('sha256') and sha256(path)!=item['sha256']:errors.append(f'SHA256 mismatch {item["path"]}');continue
   valid.append(item)
  entry['local_verification']={'checked_files':len(valid),'bytes':sum((ROOT/f['path']).stat().st_size for f in valid),'errors':errors,'sha256_checked':verify}
  entry.setdefault('training_admitted',entry.get('training_approved',entry.get('training_eligible',False)))
  license_checks=[]
  if entry['training_admitted']:
   # Each group stores exact evidence receipts under its source directory.
   candidates=[ROOT/'data/decision-v1-text/licenses'/entry['source']]
   candidates += [ROOT/'data/decision-v1-text/licenses'/alias for alias,target in ALIASES.items() if target==entry['source']]
   for directory in candidates:
    for receipt in directory.glob('*.receipt.json'):
     try:
      meta=json.loads(receipt.read_text())
      evidence=receipt.with_name(receipt.name.removesuffix('.receipt.json'))
      license_checks.append(verified_license(evidence,meta['spdx']))
     except (ValueError,KeyError,OSError):continue
   entry['verified_license_evidence']=license_checks
   if not license_checks:entry['admission_blocker']='No valid allowlisted license receipt'
   if not verify:entry['admission_blocker']='SHA256 verification required'
   if any(not f.get('sha256') for f in files):entry['admission_blocker']='Missing file digest'
   if entry.get('errors') or entry.get('failures') or errors or not valid:
    entry['admission_blocker']='Acquisition or integrity errors'
   if entry.get('admission_blocker'):entry['training_admitted']=False
  if 'acquisition_status' not in entry:
   entry['acquisition_status']='partial_or_blocked' if entry.get('errors') or entry.get('failures') or errors else 'downloaded' if valid else 'not_acquired'
 return {'schema_version':1,'expected_public_sources':len(EXPECTED),'sources':[sources[n] for n in EXPECTED],
         'generation_work':[{'source':'synthetic_blind_agreement','status':'not_generated','reason':'Not a downloadable dataset; needs new generation, independent relabeling and agreement filtering.'},
                            {'source':'ticket_invoice_templates','status':'typed_decisions_workflows_acquired_if_available','reason':'The separate generated-template expansion has not been produced.'}]}

def main():
 p=argparse.ArgumentParser();p.add_argument('inventories',type=Path,nargs='+');p.add_argument('--output',type=Path,default=ROOT/'data/decision-v1-text/source-registry.json');p.add_argument('--verify',action='store_true');a=p.parse_args()
 report=consolidate(a.inventories,a.verify);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'sources':len(report['sources']),'files':sum(r['local_verification']['checked_files'] for r in report['sources']),'bytes':sum(r['local_verification']['bytes'] for r in report['sources']),'integrity_errors':sum(len(r['local_verification']['errors']) for r in report['sources'])}))
 if any(r['local_verification']['errors'] for r in report['sources']):raise SystemExit(1)

if __name__=='__main__':main()
