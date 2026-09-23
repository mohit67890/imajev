"""One fresh Codex CLI process and one or two directly attached images per item."""
import argparse,concurrent.futures,json,re,shutil,subprocess,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from imajev_bench.runner import digest,file_digest,domain

PROMPT='''Answer the single image-and-text decision below using only the attached images and supplied request. Image 1 and Image 2 mean attachment order when two images are present. Inspect the attached pixels directly. Apply the complete rule and field descriptions. Do not use tools, read files, browse, or inspect any other case. Return the exact typed value, or null when the supplied evidence cannot determine it. Give a brief visible-evidence and rule justification. Do not assume missing facts.\n\n'''


def validate_item(item):
    if not isinstance(item.get('id'),str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',item['id']):raise ValueError('Unsafe item ID')
    if len(item['request']['fields'])!=1 or not 1<=len(item['images'])<=2:raise ValueError('Exactly one field and one or two images required')
    if item['request']['request_id']!=item['id']:raise ValueError('Request ID mismatch')
    if digest({'request':item['request'],'images':item['images']})!=item['input_sha256']:raise ValueError('Input hash mismatch')


def run_one(item,asset_root,output,model):
    validate_item(item)
    rid=item['id'];target=output/rid;target.mkdir(exist_ok=False)
    if digest({'request':item['request'],'images':item['images']})!=item['input_sha256']:raise ValueError('Input hash mismatch')
    schema={'type':'object','properties':{'value':{'enum':[v for _,v in domain(item['request']['fields'][0])]+[None]},'evidence':{'type':'string'}},'required':['value','evidence'],'additionalProperties':False}
    (target/'response-schema.json').write_text(json.dumps(schema))
    prompt=PROMPT+json.dumps(item['request'],ensure_ascii=False)
    (target/'prompt.txt').write_text(prompt)
    with tempfile.TemporaryDirectory(prefix='imajev-single-') as tmp:
        work=Path(tmp);images=[]
        for i,asset in enumerate(item['images']):
            src=(asset_root/asset['path']).resolve()
            if not src.is_relative_to(asset_root.resolve()) or file_digest(src)!=asset['sha256']:raise ValueError('Image path/hash mismatch')
            dst=work/f'image-{i}{src.suffix}';shutil.copyfile(src,dst);images.extend(['--image',str(dst)])
        command=['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','--sandbox','read-only','--model',model,'-c','model_reasoning_effort="medium"','--json','--output-schema',str((target/'response-schema.json').resolve()),'--output-last-message',str((target/'answer.json').resolve()),'--cd',str(work),*images,'-']
        start=time.monotonic()
        with (target/'events.jsonl').open('x') as stdout,(target/'stderr.txt').open('x') as stderr:
            proc=subprocess.run(command,input=prompt,text=True,stdout=stdout,stderr=stderr,timeout=300)
        receipt={'id':rid,'input_sha256':item['input_sha256'],'image_sha256_in_order':[asset['sha256'] for asset in item['images']],'model_requested':model,'reasoning_effort':'medium','elapsed_seconds':time.monotonic()-start,'returncode':proc.returncode,'command':command,'prompt_sha256':file_digest(target/'prompt.txt'),'events_sha256':file_digest(target/'events.jsonl'),'tool_items':[]}
        for line in (target/'events.jsonl').read_text().splitlines():
            event=json.loads(line);entry=event.get('item',{})
            if entry and entry.get('type') not in ('agent_message','reasoning'):receipt['tool_items'].append(entry.get('type','unknown'))
            if event.get('type')=='thread.started':receipt['thread_id']=event.get('thread_id')
            if event.get('type')=='turn.completed':receipt['usage']=event.get('usage')
        receipt['status']='failed'
        if proc.returncode==0 and (target/'answer.json').exists() and not receipt['tool_items']:
            answer=json.loads((target/'answer.json').read_text());allowed=[v for _,v in domain(item['request']['fields'][0])]+[None]
            if any(type(v) is type(answer.get('value')) and v==answer['value'] for v in allowed) and isinstance(answer.get('evidence'),str) and answer['evidence'].strip():
                prediction={'id':rid,'input_sha256':item['input_sha256'],**answer};(target/'prediction.json').write_text(json.dumps(prediction)+'\n');receipt['status']='complete';receipt['prediction_sha256']=file_digest(target/'prediction.json')
        (target/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({'id':rid,'status':receipt['status'],'returncode':proc.returncode}),flush=True)
        return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);p.add_argument('--workers',type=int,default=3);p.add_argument('--resume',action='store_true');a=p.parse_args()
    packet=json.loads(a.packet.read_text());model='gpt-5.6-sol';config={'packet_sha256':file_digest(a.packet),'model':model,'runner_sha256':file_digest(Path(__file__)),'reasoning_effort':'medium','cli_version':subprocess.check_output(['codex','--version'],text=True).strip(),'workers':a.workers,'prompt_prefix':PROMPT}
    if not 1<=a.workers<=3 or (a.limit is not None and a.limit<1):raise ValueError('Use 1–3 workers and a positive limit')
    if not Path(packet['asset_root']).is_absolute():raise ValueError('Asset root must be absolute')
    if len({i['id'] for i in packet['items']})!=len(packet['items']):raise ValueError('Duplicate IDs')
    for item in packet['items']:validate_item(item)
    if a.resume:
        old=json.loads((a.output/'config.json').read_text())
        if old!=config:raise ValueError('Resume configuration changed')
    else:
        a.output.mkdir(parents=True,exist_ok=False);(a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    pending=[]
    for item in packet['items']:
        target=a.output/item['id']
        if target.exists():
            receipt=json.loads((target/'receipt.json').read_text())
            if receipt['status']!='complete' or receipt['input_sha256']!=item['input_sha256'] or file_digest(target/'prediction.json')!=receipt['prediction_sha256']:raise ValueError('Existing failed or changed attempt; use an explicit new run')
            if file_digest(target/'events.jsonl')!=receipt['events_sha256'] or file_digest(target/'prompt.txt')!=receipt['prompt_sha256']:raise ValueError('Existing raw artifact changed')
        else:pending.append(item)
    if a.limit is not None:pending=pending[:a.limit]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as executor:
        receipts=list(executor.map(lambda item:run_one(item,Path(packet['asset_root']),a.output,model),pending))
    if any(r['status']!='complete' for r in receipts):raise SystemExit('One or more attempts failed; raw diagnostics retained')
    paths=[a.output/i['id']/'prediction.json' for i in packet['items']]
    if all(p.exists() for p in paths):
        combined={'protocol':'imajev-modality-probe-v1','model_name':model,'predictions':[json.loads(p.read_text()) for p in paths]}
        with (a.output/'predictions.json').open('x') as f:json.dump(combined,f,indent=2);f.write('\n')
        print('All items complete',flush=True)
if __name__=='__main__':main()
