"""Verify observed CLI isolation and compare paired predictions without causal claims."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from imajev_bench.runner import file_digest,digest
from imajev_bench.schema import validate_records,model_payload
from analyze_ablations import validate_predictions,equal


def audit(packet,run):
    config=json.loads((run/'config.json').read_text());threads=set();receipts=[];files=[];usage={}
    if file_digest(run/'runner-used.py')!=config['runner_sha256']:raise ValueError('Runner snapshot changed')
    for item in packet['items']:
        rid=item['id'];target=run/rid;r=json.loads((target/'receipt.json').read_text());command=r['command']
        if r['status']!='complete' or r['returncode']!=0 or r['input_sha256']!=item['input_sha256']:raise ValueError('Incomplete/mismatched attempt')
        for name,key in [('events.jsonl','events_sha256'),('prompt.txt','prompt_sha256'),('prediction.json','prediction_sha256')]:
            if file_digest(target/name)!=r[key]:raise ValueError('Receipt artifact changed')
        if (target/'prompt.txt').read_text()!=config['prompt_prefix']+json.dumps(item['request'],ensure_ascii=False):raise ValueError('Prompt contains unexpected content')
        if command.count('--image')!=len(item['images']) or '--ephemeral' not in command or '--ignore-user-config' not in command or command[command.index('--sandbox')+1]!='read-only':raise ValueError('Unexpected CLI isolation arguments')
        if 'image_sha256_in_order' in r and r['image_sha256_in_order']!=[asset['sha256'] for asset in item['images']]:raise ValueError('Image order mismatch')
        started=[]
        for line in (target/'events.jsonl').read_text().splitlines():
            e=json.loads(line)
            if e.get('type')=='thread.started':started.append(e['thread_id'])
            if e.get('item') and e['item'].get('type') not in ('agent_message','reasoning'):raise ValueError('Tool or unknown item in accepted run')
        if len(started)!=1 or started[0] in threads or started[0]!=r['thread_id']:raise ValueError('Context ID reused/missing')
        threads.add(started[0]);answer=json.loads((target/'answer.json').read_text());prediction=json.loads((target/'prediction.json').read_text())
        if not equal(answer['value'],prediction['value']) or answer['evidence']!=prediction['evidence']:raise ValueError('Raw answer differs from prediction')
        receipts.append({'id':rid,'thread_id':started[0],'input_sha256':r['input_sha256'],'elapsed_seconds':r['elapsed_seconds']})
        for k,v in r.get('usage',{}).items():usage[k]=usage.get(k,0)+v
        files.extend({'path':str(p),'sha256':file_digest(p)} for p in sorted(target.iterdir()) if p.is_file())
    return {'unique_contexts':len(threads),'observed_tool_items':0,'direct_image_count_by_id':{i['id']:len(i['images']) for i in packet['items']},'usage_reported_by_cli':usage,'receipts':receipts,'files':files,'limits':'Event-based audit, not OS filesystem confinement. Requested model identity is not independent served-build verification.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    packetpath=a.root/'blind.json';packet=json.loads(packetpath.read_text());config=json.loads((a.run/'config.json').read_text())
    if file_digest(packetpath)!=config['packet_sha256']:raise ValueError('Packet changed')
    result=audit(packet,a.run)
    rows=validate_records([json.loads(l) for l in (a.root/'records.jsonl').read_text().splitlines()],a.root)
    for row in rows:
        item=next(i for i in packet['items'] if i['id']==row['id'])
        if item['input_sha256']!=digest(model_payload(row)):raise ValueError('Packet/record mismatch')
    before=validate_predictions(packet,json.loads(a.prior.read_text()));after=validate_predictions(packet,json.loads((a.run/'predictions.json').read_text()))
    comparison=[]
    for row in rows:
        rid=row['id'];old=equal(before[rid]['value'],row['gold']);new=equal(after[rid]['value'],row['gold']);comparison.append({'id':rid,'prior_value':before[rid]['value'],'isolated_value':after[rid]['value'],'reference':row['gold'],'prior_correct':old,'isolated_correct':new})
    result['comparison']={'items':len(rows),'prior_matches':sum(r['prior_correct'] for r in comparison),'isolated_matches':sum(r['isolated_correct'] for r in comparison),'corrected':sum(not r['prior_correct'] and r['isolated_correct'] for r in comparison),'regressed':sum(r['prior_correct'] and not r['isolated_correct'] for r in comparison),'per_item':comparison}
    result['source_files']=[{'path':str(p),'sha256':file_digest(p)} for p in [packetpath,a.root/'records.jsonl',a.prior,a.run/'predictions.json',a.run/'config.json',a.run/'runner-used.py']]
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'contexts':result['unique_contexts'],'tools':0,'comparison':{k:v for k,v in result['comparison'].items() if k!='per_item'},'usage':result['usage_reported_by_cli']},indent=2))
