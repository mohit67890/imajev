"""Validate complete model reviews and report consensus without promoting human gold."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from imajev_bench.schema import validate_records,model_payload
from imajev_bench.runner import digest


def compare(records, exports, root):
    records=validate_records(records,root)
    indexed={r['id']:r for r in records}
    expected=hashlib.sha256(json.dumps(sorted((r['id'],digest(model_payload(r))) for r in records),separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    reviewers=set();answers=[];flags=set();identities=[]
    for ex in exports:
        if ex.get('annotator_type')!='model' or ex.get('purpose')!='independent_review' or ex.get('format_version')!='0.0.1' or ex.get('protocol')!='imajev-bench-blind-review-v1':raise ValueError('Expected independent model review protocol')
        reviewer=ex.get('reviewer_id')
        if not isinstance(reviewer,str) or not reviewer.strip() or reviewer in reviewers:raise ValueError('Distinct reviewer IDs required')
        reviewers.add(reviewer)
        if ex.get('input_sha256')!=expected:raise ValueError('Dataset hash mismatch')
        seen=set();values={}
        for kind,note in [('reviews','evidence'),('flags','reason')]:
            for item in ex[kind]:
                rid=item['id']
                if rid not in indexed or rid in seen:raise ValueError('Unknown or duplicate ID')
                seen.add(rid);row=indexed[rid]
                if item.get('reviewer_id')!=reviewer or item.get('input_sha256')!=digest(model_payload(row)):raise ValueError('Reviewer or item hash mismatch')
                if not isinstance(item.get(note),str) or not item[note].strip():raise ValueError('Missing evidence/reason')
                if kind=='reviews':
                    validate_records([{**row,'gold':item['value'],'annotation_status':'draft'}],root)
                    values[rid]=item['value']
                else:flags.add(rid)
        if seen!=set(indexed):raise ValueError('Incomplete review')
        answers.append(values);identities.append({'reviewer_id':reviewer,'model_name_as_reported':ex.get('model_name'),'judged':len(values),'unknown':sum(v is None for v in values.values()),'flagged':len(ex['flags'])})
    if len(exports)<2:raise ValueError('At least two reviews required')
    consensus=[];disagreements=[]
    for rid,row in indexed.items():
        if rid in flags:continue
        values=[a[rid] for a in answers]
        if all(type(v) is type(values[0]) and v==values[0] for v in values):
            consensus.append({'id':rid,'value':values[0],'matches_provisional':type(values[0]) is type(row['gold']) and values[0]==row['gold']})
        else:disagreements.append({'id':rid,'values':dict(zip([i['reviewer_id'] for i in identities],values))})
    unresolved=set(flags)|{d['id'] for d in disagreements}
    groups=sorted({indexed[r]['group_id'] for r in unresolved})
    return {'status':'model_consensus_only','dataset_input_sha256':expected,'reviewers':identities,'records':len(records),'consensus_count':len(consensus),'consensus':consensus,'disagreements':disagreements,'flagged_ids':sorted(flags),'groups_requiring_review':groups,'human_validation':False,'limitations':'Reported model identities and blindness are not independently verifiable. Agreement does not establish accuracy. Canonical labels and review status are unchanged.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--reviews',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(l) for l in a.records.read_text().splitlines()]
    result=compare(rows,[json.loads(p.read_text()) for p in a.reviews],a.records.parent)
    result['source_files']=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [a.records,*a.reviews]]
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='consensus'},indent=2))
if __name__=='__main__':main()
