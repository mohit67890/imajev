"""Create paired, gold-free modality diagnostics from a draft development dataset."""
import argparse,copy,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from imajev_bench.schema import validate_records,model_payload
from imajev_bench.runner import digest,file_digest


def build(rows,root,output):
    rows=validate_records(rows,root);output.mkdir(parents=True,exist_ok=False)
    eligible=[r for r in rows if r['images']]
    manifest={'status':'development_modality_diagnostic','conditions':{},'reference_label_status':'provisional with two-model consensus; not human validated','ablation_gold_status':'not independently annotated; no ablated accuracy claim'}
    for condition in ('full','no_image','no_state'):
        items=[];mapping=[]
        for r in eligible:
            if condition=='no_state' and r['track']!='joint':continue
            payload=copy.deepcopy(model_payload(r))
            if condition=='no_image':payload['images']=[]
            if condition=='no_state':payload['request']['state']={}
            # Stable IDs join conditions without encoding original source/group/labels.
            rid='probe-'+digest(r['id'])[:12];payload['request']['request_id']=rid
            items.append({'id':rid,'input_sha256':digest(payload),'request':payload['request'],'images':payload['images']})
            mapping.append({'id':rid,'original_id':r['id'],'group_id':r['group_id'],'track':r['track'],'family':r['family'],'full_reference':r['gold'],'input_sha256':digest(payload)})
        random.Random(20260922).shuffle(items)
        path=output/(condition+'.json')
        path.write_text(json.dumps({'protocol':'imajev-modality-probe-v1','asset_root':str(root.resolve()),'items':items},indent=2)+'\n')
        manifest['conditions'][condition]={'packet':str(path),'packet_sha256':file_digest(path),'count':len(items),'mapping':mapping}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=build([json.loads(l) for l in a.records.read_text().splitlines()],a.records.parent,a.output)
    print({k:v['count'] for k,v in result['conditions'].items()})
