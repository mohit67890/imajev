"""Paired diagnostic summaries; never treat original gold as ablated answerability gold."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from imajev_bench.runner import digest,file_digest,domain


def equal(a,b):return type(a) is type(b) and a==b


def validate_predictions(packet, export):
    if export.get('protocol')!='imajev-modality-probe-v1':raise ValueError('Prediction protocol mismatch')
    byid={r['id']:r for r in packet['items']};out={}
    for p in export['predictions']:
        rid=p['id']
        if rid not in byid or rid in out:raise ValueError('Duplicate or unknown prediction ID')
        row=byid[rid]
        if p.get('input_sha256')!=row['input_sha256']:raise ValueError('Prediction input hash mismatch')
        if digest({'request':row['request'],'images':row['images']})!=row['input_sha256']:raise ValueError('Packet payload hash mismatch')
        if 'value' not in p or not any(equal(p['value'],v) for _,v in domain(row['request']['fields'][0])+[('__unknown__',None)]):raise ValueError('Invalid typed decision')
        if not isinstance(p.get('evidence'),str) or not p['evidence'].strip():raise ValueError('Missing prediction evidence')
        out[rid]=p
    if set(out)!=set(byid):raise ValueError('Incomplete predictions')
    return out


def summarize(manifest, packets, exports):
    predictions={k:validate_predictions(packets[k],exports[k]) for k in manifest['conditions']}
    result={'status':'exploratory_model_modality_diagnostic','conditions':{},'paired':{},'limitations':[
        'One isolated agent run per condition; no fixed inference seed, pinned endpoint, token budget or latency measurement.',
        'Sol participated in drafting and annotation; fresh contexts avoid direct label exposure but not model-family bias.',
        'Full labels have model consensus, not human validation. Ablated labels were not independently annotated.',
        'No-state removes all structured state (policy and instance parameters), retaining image and question/options; it is not a pure text-free or policy-only intervention.',
        'Unknown rates and retention against full-input labels are diagnostics, not ablated accuracy or a benchmark ranking.',
        'Small public development batch, repeated templates and 11 scenes; no generalization or confidence-interval claim.']}
    full=predictions['full']
    for condition,info in manifest['conditions'].items():
        rows=info['mapping'];pred=predictions[condition]
        strata={}
        for name,selected in [('all',rows),('full_answerable',[r for r in rows if r['full_reference'] is not None]),('full_unknown',[r for r in rows if r['full_reference'] is None]),('joint',[r for r in rows if r['track']=='joint']),('visual_factual',[r for r in rows if r['family']=='visible_evidence'])]:
            if not selected:continue
            n=len(selected);matches=sum(equal(pred[r['id']]['value'],r['full_reference']) for r in selected)
            strata[name]={'n':n,'unknown':sum(pred[r['id']]['value'] is None for r in selected),'full_reference_matches':matches,'full_reference_match_rate':matches/n}
        result['conditions'][condition]={'model_name_as_reported':exports[condition].get('model_name'),'strata':strata}
        if condition!='full':
            known=[r for r in rows if r['full_reference'] is not None]
            correct=[r for r in known if equal(full[r['id']]['value'],r['full_reference'])]
            result['paired'][condition]={'full_answerable_n':len(known),'full_correct_n':len(correct),'full_correct_to_unknown':sum(pred[r['id']]['value'] is None for r in correct),'full_correct_answer_retained':sum(equal(pred[r['id']]['value'],r['full_reference']) for r in correct),'changed_value_on_full_answerable':sum(not equal(full[r['id']]['value'],pred[r['id']]['value']) for r in known)}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--full',type=Path,required=True);p.add_argument('--no-image',type=Path,required=True);p.add_argument('--no-state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());packets={};sources=[a.manifest]
    for key,info in manifest['conditions'].items():
        path=Path(info['packet'])
        if file_digest(path)!=info['packet_sha256']:raise ValueError('Packet file changed')
        packets[key]=json.loads(path.read_text());sources.append(path)
    paths={'full':a.full,'no_image':a.no_image,'no_state':a.no_state};exports={k:json.loads(p.read_text()) for k,p in paths.items()}
    result=summarize(manifest,packets,exports);sources.extend(paths.values());result['source_files']=[{'path':str(p),'sha256':file_digest(p)} for p in sources]
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_files'},indent=2))
if __name__=='__main__':main()
