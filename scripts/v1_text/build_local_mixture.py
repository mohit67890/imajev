"""Build the audited v1.1 mixture: 200k text decisions + up to 300k v1 image decisions.

Image admission follows policy A (see admit_v1_image_sources.py) when its report exists;
otherwise the strict primary-grant migration (abo + vizwiz only) is used.
"""
from __future__ import annotations
import collections, json, hashlib, os
from pathlib import Path
from .common import verified_license, write_jsonl
from .assemble_mixture import stratified_train
from .build_controls import add_controls, grounded_contradictions
from .audit_mixture import audit
from .augment_text import augment, reference_controls
from .deduplicate_text import deduplicate
ROOT=Path(__file__).resolve().parents[2]
IMAGE_LIMIT=300000
CALIBRATION_PERCENT=0  # image calibration is fitted on the image dev partition instead: a per-image fold split whole-category groups (defects) and multi-photo listings (abo)

def relative_license(lic):
 """Rewrite absolute evidence/receipt paths to repo-relative ones (same files, pod-portable)."""
 out=dict(lic)
 for key in ('evidence','receipt'):
  value=out.get(key)
  if isinstance(value,str) and value.startswith(str(ROOT)+os.sep):out[key]=os.path.relpath(value,ROOT)
 return out

def image_mapping(data):
 """(source -> licence string -> licence object); state_aware resolved via its origin prefix."""
 policy=ROOT/'reports/v1.1-datasets/image-policy-a.json'
 if policy.exists():
  report=json.loads(policy.read_text());mapping={}
  for source,by_string in report['mapping_by_source'].items():
   mapping[source]={ls:verified_license(Path(info['evidence']),info['spdx']) for ls,info in by_string.items()}
  state={origin:{ls:verified_license(Path(info['evidence']),info['spdx']) for ls,info in by_string.items()} for origin,by_string in report['state_aware_by_origin'].items()}
  return mapping,state,'A: v1 image posture'
 lic=verified_license(Path('data/decision-v1-text/licenses/abo/LICENSE.txt'),'CC-BY-4.0');mapping={'abo':{'CC-BY-4.0':lic}}
 migration=ROOT/'reports/v1.1-datasets/image-license-migration.json'
 if migration.exists():
  for source,item in json.loads(migration.read_text()).get('mapping_by_source',{}).items():
   info=item['license'];mapping[source]={'CC-BY-4.0':verified_license(Path(info['evidence']),info['spdx'])}
 return mapping,{},'strict primary grant'

def image_license(r,mapping,state):
 if r['source']=='state_aware':
  origin=r['source_group'].split(':',1)[0];return state.get(origin,{}).get(r['license'])
 return mapping.get(r['source'],{}).get(r['license'])

def calibration_hash(sha,seed='v1.1:image-calibration'):
 return int(hashlib.sha256(f"{seed}\0{sha}".encode()).hexdigest()[:8],16)%100

def main():
 os.chdir(ROOT)
 data=ROOT/'data/decision-v1-text';paths=sorted((data/'converted').glob('*.jsonl'))
 text=[json.loads(l) for p in paths for l in p.open() if l.strip()]
 for r in text:r['license']=relative_license(r['license'])
 exclusion_path=ROOT/'reports/v1.1-datasets/overlength-exclusions.json'
 overlength=set(json.loads(exclusion_path.read_text())['record_ids']) if exclusion_path.exists() else set()
 text=[r for r in text if r['id'] not in overlength]
 text,dedup_removed=deduplicate(text)
 mapping,state,policy=image_mapping(data)
 text_keys={(r.get('source'),r.get('source_group')) for r in text}  # e.g. ABO listings also exist as text records
 images=[];excluded=collections.Counter()
 for line in (ROOT/'data/manifests/decision-v1.jsonl').open():
  r=json.loads(line);lic=image_license(r,mapping,state)
  if lic is None:excluded[r['source']]+=1;continue
  # v1 re-partitioned VSR by image; its upstream test items are not carried into v1.1 training or dev.
  if str(r.get('source_split','')).lower()=='test' and r['partition']!='test':excluded[r['source']+':upstream_test_split']+=1;continue
  r['license_string']=r['license'];r['license']=lic
  # Calibration fold: never trained on; assigned per image hash so photos shared across sources agree.
  if r['partition']=='train' and (r['source'],r['source_group']) not in text_keys:
   flags={calibration_hash(im['sha256'])<CALIBRATION_PERCENT for im in r['images']}
   if flags=={True}:r['partition']='calibration'
   elif len(flags)>1:excluded[r['source']+':mixed_calibration_pair']+=1;continue
  images.append(r)
 def index(rows):
  parts=collections.defaultdict(set);image_parts=collections.defaultdict(set)
  for r in rows:
   parts[(r['source'],r['source_group'])].add(r['partition'])
   for im in r['images']:image_parts[im['sha256']].add(r['partition'])
  return parts,image_parts
 # v1 allowed a group (e.g. a defects category or a multi-photo ABO listing) to span train and dev; v1.1 does not.
 # Rather than dropping the whole group, dev records that share a group or image with train become train.
 parts,image_parts=index(images);moved=collections.Counter();moved_keys=set()
 for r in images:
  if r['partition']!='dev':continue
  touched=parts[(r['source'],r['source_group'])]|set().union(*(image_parts[im['sha256']] for im in r['images']))
  if 'train' in touched and 'test' not in touched:r['partition']='train';moved[r['source']]+=1;moved_keys.add((r['source'],r['source_group']))
 for r in text:
  if r.get('partition')=='dev' and (r.get('source'),r.get('source_group')) in moved_keys:r['partition']='train';moved[r['source']+':text']+=1
 parts,image_parts=index(images)
 kept=[];dropped=collections.Counter()
 for r in images:
  if len(parts[(r['source'],r['source_group'])])==1 and all(len(image_parts[im['sha256']])==1 for im in r['images']):kept.append(r)
  else:dropped[r['source']]+=1
 images=kept;available=collections.Counter(r['source'] for r in images if r['partition']=='train')
 text=stratified_train(augment(text)+reference_controls(text),200000,'v1.1:text');images=stratified_train(images,IMAGE_LIMIT,'v1.1:image')
 rows=add_controls(text,images);contradictions=grounded_contradictions(images);rows+=contradictions
 result=audit(rows)
 result['overlength_excluded_record_ids']=sorted(overlength)
 result['exact_state_overlap_removed']=dedup_removed
 result['input_files']=[str(p.relative_to(ROOT)) for p in paths]
 result['image_policy']=policy
 result['excluded_image_records']=dict(excluded)
 result['partition_inconsistent_image_records_dropped']=dict(dropped)
 result['dev_records_moved_to_train_for_group_consistency']=dict(moved)
 result['image_train_records_available_by_source']=dict(available)
 result['image_train_records_selected_by_source']=dict(collections.Counter(r['source'] for r in images if r['partition']=='train'))
 result['contradiction_decisions']=sum(len(r['request']['fields']) for r in contradictions)
 result['training_manifest_decisions']=sum(len(r['request']['fields']) for r in rows if r['partition']=='train')
 result['train_decisions']={kind:sum(len(r['request']['fields']) for r in collection if r['partition']=='train') for kind,collection in [('text',text),('image',images)]}
 result['status']='audited_partial_mixture' if result['ok'] else 'audit_failed'
 result['remaining']=['Synthetic slice dropped for v1.1 (pilots failed blind relabel).','Full processed-token audit and training acceptance gates remain required.']
 out=ROOT/'reports/v1.1-datasets/mixture-audit.json';out.write_text(json.dumps(result,indent=2)+'\n')
 if not result['ok']:
  print(json.dumps({'ok':False,'errors':result['errors'][:20],'error_count':len(result['errors'])}));raise SystemExit(1)
 destination=ROOT/'data/manifests/decision-v1.1-draft.jsonl';write_jsonl(destination,rows)
 result['manifest_sha256']=hashlib.sha256(destination.read_bytes()).hexdigest();result['records']=len(rows)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('counts','licenses','errors','excluded_image_records')}))
if __name__=='__main__':main()
