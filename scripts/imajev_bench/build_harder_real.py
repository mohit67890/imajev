"""Build higher-complexity development requests over previously acquired licensed photos."""
import argparse,copy,hashlib,json,random,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from imajev_bench.schema import validate_records,model_payload
from imajev_bench.runner import digest,file_digest
from imajev_bench.audit import audit


def build(source_root,case_files,output):
    sources={s['source']['source_id']:s for s in json.loads((source_root/'collection.json').read_text())['sources']}
    cases=[]
    for path in case_files:cases.extend(json.loads(path.read_text()))
    parent={sid:sid for c in cases for sid in c['source_ids']}
    def find(s):
        while parent[s]!=s:s=parent[s]
        return s
    for c in cases:
        if not 1<=len(c['source_ids'])<=2 or len(set(c['source_ids']))!=len(c['source_ids']):raise ValueError('Expected one or two distinct sources')
        for sid in c['source_ids']:
            if sid not in sources:raise ValueError('Unknown source')
            parent[find(sid)]=find(c['source_ids'][0])
    members={}
    for sid in parent:members.setdefault(find(sid),[]).append(sid)
    gids={sid:'evidence-'+digest(sorted(members[find(sid)]))[:12] for sid in parent}
    output.mkdir(parents=True,exist_ok=False);(output/'assets').mkdir();rows=[]
    for sid in parent:
        asset=sources[sid]['local_image'];src=source_root/asset['path']
        if file_digest(src)!=asset['sha256']:raise ValueError('Source image changed')
        shutil.copyfile(src,output/asset['path'])
    for index,c in enumerate(cases):
        rid='hard-'+hashlib.sha256(f'real-v1:{index}'.encode()).hexdigest()[:12]
        field=copy.deepcopy(c['field'])
        if field['type']=='choice':random.Random(719+index).shuffle(field['options'])
        credits=[{**sources[sid]['source'],'source_page':sources[sid]['source']['file_page']} for sid in c['source_ids']]
        rows.append({'id':rid,'group_id':gids[c['source_ids'][0]],'track':'joint','family':c['family'],'split':'dev','images':[sources[sid]['local_image'] for sid in c['source_ids']],
                     'request':{'request_id':rid,'state':c['state'],'fields':[field]},'gold':c['gold'],'annotation_status':'draft','provenance':{'source_ids':c['source_ids'],'image_sources':credits,'draft_evidence':c['evidence'],'label_origin':'Sol-authored provisional label; no human approval','images_reused_from':'real-pilot-v1','synthetic':False,'rule_synthetic':True}})
    rows=validate_records(rows,output)
    (output/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    packet=[{'id':r['id'],'input_sha256':digest(model_payload(r)),**model_payload(r)} for r in rows];random.Random(8921).shuffle(packet)
    (output/'blind.json').write_text(json.dumps({'protocol':'imajev-modality-probe-v1','asset_root':str(output.resolve()),'items':packet},indent=2)+'\n')
    manifest={'status':'development_candidates_not_hidden','records_sha256':file_digest(output/'records.jsonl'),'source_records_sha256':file_digest(source_root/'records.jsonl'),'builder_sha256':file_digest(Path(__file__)),'case_files':[{'path':str(p),'sha256':file_digest(p)} for p in case_files],'audit':audit(rows),'two_image_cases':sum(len(r['images'])==2 for r in rows),'evidence_components':{gids[sids[0]]:sorted(sids) for sids in members.values()},'limitation':'All photos reused from earlier development material. Grouping joins every reused image and cross-image pair; no new independent scenes, calibration or test data.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    attribution=['# Image attributions','', 'Previously acquired photographs reused without new pixel edits. Existing orientation/resize/JPEG conversion and metadata removal described in source collection. Original per-file licenses continue to apply.','']
    for sid in sorted(parent):
        s=sources[sid]['source'];attribution.append(f"- [{sid}]({s['file_page']}) — {s['creator']}; [{s['license']}]({s['license_url']}).")
    (output/'ATTRIBUTION.md').write_text('\n'.join(attribution)+'\n')
    return rows,manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--cases',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r,m=build(a.source_root,a.cases,a.output);print(json.dumps({'records':len(r),'two_images':m['two_image_cases'],'audit':m['audit']},indent=2))
