"""Validate image triplets and report exact paired behavior, without independence claims."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from imajev_bench.schema import validate_records,model_payload
from imajev_bench.runner import digest,file_digest
from analyze_ablations import validate_predictions,equal


def check_groups(rows,manifest):
    byid={r['id']:r for r in rows};seen=set()
    for g in manifest['groups']:
        ids=[g[k] for k in ('base','flip','invariant')]
        if len(set(ids))!=3 or any(i in seen for i in ids):raise ValueError('Triplet membership duplicated')
        seen.update(ids);a,b,c=[byid[i] for i in ids]
        if any(r['group_id']!=g['group_id'] for r in (a,b,c)):raise ValueError('Group mismatch')
        requests=[{k:v for k,v in r['request'].items() if k!='request_id'} for r in (a,b,c)]
        if requests[0]!=requests[1] or requests[0]!=requests[2]:raise ValueError('Nonimage inputs changed')
        if len({r['images'][0]['sha256'] for r in (a,b,c)})!=3:raise ValueError('Image intervention absent')
        if equal(a['gold'],b['gold']) or not equal(a['gold'],c['gold']):raise ValueError('Invalid expected relation')
    if seen!=set(byid):raise ValueError('Triplets do not cover dataset')


def summarize(rows,manifest,export):
    check_groups(rows,manifest);byid={r['id']:r for r in rows}
    packet={'items':[{'id':r['id'],'input_sha256':digest(model_payload(r)),**model_payload(r)} for r in rows]}
    pred=validate_predictions(packet,export);groups=[]
    for g in manifest['groups']:
        ids=[g[k] for k in ('base','flip','invariant')];vals=[pred[i]['value'] for i in ids];correct=[equal(v,byid[i]['gold']) for i,v in zip(ids,vals)]
        groups.append({'group_id':g['group_id'],'family':g['family'],'predictions':vals,'references':[byid[i]['gold'] for i in ids],'relevant_changed':not equal(vals[0],vals[1]),'irrelevant_unchanged':equal(vals[0],vals[2]),'relevant_pair_correct':correct[0] and correct[1],'irrelevant_pair_correct':correct[0] and correct[2],'all_three_correct':all(correct)})
    def aggregate(gs):return {'groups':len(gs),**{key:sum(g[key] for g in gs) for key in ('relevant_changed','irrelevant_unchanged','relevant_pair_correct','irrelevant_pair_correct','all_three_correct')}}
    return {'status':'synthetic_development_model_diagnostic','model_name_as_reported':export.get('model_name'),'record_reference_matches':sum(equal(pred[r['id']]['value'],r['gold']) for r in rows),'records':len(rows),'unknown_outputs':sum(p['value'] is None for p in pred.values()),'overall':aggregate(groups),'by_family':{f:aggregate([g for g in groups if g['family']==f]) for f in sorted({g['family'] for g in groups})},'group_results':groups,'human_validated':False,'limitation':'12 groups share 3 templates. Execution-context isolation is not inferred from predictions; consult run receipts. No independent-trial, served-build identity or real-photo robustness claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    mp=a.root/'manifest.json';rp=a.root/'records.jsonl';manifest=json.loads(mp.read_text())
    if file_digest(rp)!=manifest['records_sha256']:raise ValueError('Records changed')
    rows=validate_records([json.loads(l) for l in rp.read_text().splitlines()],a.root)
    result=summarize(rows,manifest,json.loads(a.predictions.read_text()));result['source_files']=[{'path':str(p),'sha256':file_digest(p)} for p in (rp,mp,a.predictions,Path(__file__))]
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('group_results','source_files')},indent=2))
