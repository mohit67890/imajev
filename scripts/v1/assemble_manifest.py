"""Merge validated decision-v1 sources into one manifest, removing cross-source train/test image leaks."""
import json,hashlib,sys
from pathlib import Path
from collections import Counter
VERSION='decision-v1'
# Validated but deliberately left out: licence ruled restricted/unclear for commercial use, machine-generated train labels, and only 1.9k fetched.
EXCLUDE={'pixmo_count'}
root=Path('data/decision-v1');rows=[];skipped=[]
for d in sorted(root.iterdir()):
 v=d/'validation.json'
 if not d.is_dir() or d.name in EXCLUDE:continue
 if not v.exists() or json.loads(v.read_text())['summary']['errors']:skipped.append(d.name);continue
 for line in (d/'records.jsonl').read_text().splitlines():
  r=json.loads(line);r['group']='held_out_source' if d.name.startswith('heldout_') else 'trained_source';rows.append(r)
# ---- policy filters found by scripts/v1/audit_manifest.py (sources on disk stay untouched) ----
policy=Counter();F=lambda r:r['request']['fields'][0]
AMBIGUOUS={'unknown','unanswerable','n/a','na','not sure','unsure',"can't tell",'cannot tell','cant tell',"don't know",'dont know','not possible','unclear'}
def ambiguous(r):  # an option that reads like our own abstention line makes the right answer undefined
 return F(r)['type']=='choice' and any(str(o['value']).strip().lower() in AMBIGUOUS for o in F(r)['options'])
def vizwiz_surplus(r):  # 60% of VizWiz multiple-choice training questions were "unknown"; cap the share near 45% so photo style alone does not predict abstention
 return r['source']=='vizwiz' and r['partition']!='test' and F(r)['type']=='choice' and r['abstention_cause']=='insufficient_evidence' and int(hashlib.sha256(r['id'].encode()).hexdigest(),16)%1000>=600
SURFACES={'table','building','tree','trees','roof','floor','wall','walls','ground','grass','sky','road','water','counter','field','sidewalk','street','ceiling','fence','snow','sand','mountain','mountains','hill','bed','couch','sofa','carpet','rug','pavement','dirt','ocean','river','lake','beach','bushes','bush','hedge','platform','runway','track','tracks','court','path','lawn','background','room','kitchen','desk','shelf','shelves','clouds','cloud','leaves','hair','tile','tiles','bricks','brick'}
def unhidden(r):  # one box cannot hide a large surface: the table or the grass is still visible around the patch, so "unknown" would be a wrong label
 return r['source']=='masked_evidence' and r['target'] is None and (r.get('object_name') in SURFACES or r.get('treated_area_fraction',1)<.03)
seen=set();filtered=[]
for r in rows:
 key=(tuple(im['sha256'] for im in r['images']),F(r)['question'],F(r)['type'])
 if ambiguous(r):policy['ambiguous_unknown_like_option']+=1
 elif unhidden(r):policy['masked_evidence_surface_not_really_hidden']+=1
 elif vizwiz_surplus(r):policy['vizwiz_unknown_rebalance']+=1
 elif key in seen and not r['source'].startswith('heldout_'):policy['exact_duplicate_question']+=1
 else:filtered.append(r);seen.add(key)
rows=filtered
test_images={im['sha256'] for r in rows if r['partition']=='test' for im in r['images']}
kept=[r for r in rows if r['partition']=='test' or not any(im['sha256'] in test_images for im in r['images'])]
dropped=Counter(r['source'] for r in rows if r['partition']!='test' and any(im['sha256'] in test_images for im in r['images']))
assert len({r['id'] for r in kept})==len(kept),'duplicate ids across sources'
content=''.join(json.dumps(r,sort_keys=True)+'\n' for r in kept);out=Path(f'data/manifests/{VERSION}.jsonl');out.write_text(content)
fit=[r for r in kept if r['partition']!='test'];ftype=lambda r:r['request']['fields'][0]['type']
summary=dict(manifest_sha256=hashlib.sha256(content.encode()).hexdigest(),sources=sorted({r['source'] for r in kept}),skipped_unvalidated=skipped,excluded_by_policy=sorted(EXCLUDE),
 records=dict(Counter(r['partition'] for r in kept)),dropped_for_cross_source_test_overlap=dict(dropped),dropped_by_policy_filters=dict(policy),
 train_by_source=dict(Counter(r['source'] for r in kept if r['partition']=='train').most_common()),train_field_types=dict(Counter(ftype(r) for r in fit)),
 train_abstention=dict(Counter(str(r['abstention_cause']) for r in fit)),abstention_by_field_type={t:round(sum(r['target'] is None for r in fit if ftype(r)==t)/max(1,sum(ftype(r)==t for r in fit)),3) for t in ('choice','boolean','ordinal')},
 two_image_train=sum(len(r['images'])==2 for r in fit),soft_targets=sum('target_distribution' in r for r in fit),test_by_group=dict(Counter((r['group']) for r in kept if r['partition']=='test')),
 note='Exact-hash leak removal only; near-duplicate (re-encoded) images across sources are not detected here.')
Path('reports/decision-v1').mkdir(parents=True,exist_ok=True);Path('reports/decision-v1/manifest-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
