"""Audit a completed isolated run and describe provisional-reference agreement."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from imajev_bench.schema import validate_records,model_payload
from imajev_bench.runner import digest,file_digest
from imajev_bench.audit import audit as dataset_audit
from analyze_ablations import validate_predictions,equal
from audit_isolated_run import audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=validate_records([json.loads(l) for l in (a.root/'records.jsonl').read_text().splitlines()],a.root);packet=json.loads((a.root/'blind.json').read_text());config=json.loads((a.run/'config.json').read_text())
    if config['packet_sha256']!=file_digest(a.root/'blind.json'):raise ValueError('Packet changed')
    if {i['id'] for i in packet['items']}!={r['id'] for r in rows}:raise ValueError('Dataset/packet membership mismatch')
    lookup={r['id']:r for r in rows}
    for item in packet['items']:
        if item['input_sha256']!=digest(model_payload(lookup[item['id']])):raise ValueError('Packet/model input mismatch')
    predictions=validate_predictions(packet,json.loads((a.run/'predictions.json').read_text()));isolation=audit(packet,a.run)
    correct=lambda r:equal(predictions[r['id']]['value'],r['gold'])
    def counts(selected):return {'cases':len(selected),'reference_matches':sum(correct(r) for r in selected),'unknown_outputs':sum(predictions[r['id']]['value'] is None for r in selected)}
    mismatches=[{'id':r['id'],'reference':r['gold'],'prediction':predictions[r['id']],'draft_evidence':r['provenance'].get('draft_evidence')} for r in rows if not correct(r)]
    summary={'status':'development_diagnostic_against_provisional_labels','model_requested':config['model'],'overall':counts(rows),'single_image':counts([r for r in rows if len(r['images'])==1]),'two_image':counts([r for r in rows if len(r['images'])==2]),'reference_unknown':counts([r for r in rows if r['gold'] is None]),'reference_definite':counts([r for r in rows if r['gold'] is not None]),'by_type':{t:counts([r for r in rows if r['request']['fields'][0]['type']==t]) for t in ('boolean','choice','ordinal')},'mismatches':mismatches,'dataset_audit':dataset_audit(rows),'isolation_audit':isolation,'human_validated':False,'limitation':'Previously exposed public photos, few connected evidence groups and author/evaluator model-family overlap. Reference agreement is not established difficulty or independently verified accuracy. No CI, holdout or ranking claim.'}
    summary['source_files']=[{'path':str(f),'sha256':file_digest(f)} for f in [a.root/'records.jsonl',a.root/'manifest.json',a.root/'blind.json',a.run/'config.json',a.run/'predictions.json',Path(__file__)]]
    with a.output.open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('isolation_audit','source_files','dataset_audit')},indent=2))
if __name__=='__main__':main()
