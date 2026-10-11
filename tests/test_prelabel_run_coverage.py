import json
import pytest
from imajev_bench.api_models import prelabel_export,run_api
from imajev_bench.runner import file_digest
from imajev_bench.schema import validate_records
from test_imajev_bench_triage import record,FakeProvider

@pytest.mark.parametrize('artifact,mode',[('raw.jsonl','missing'),('raw.jsonl','duplicate'),('predictions.jsonl','missing'),('predictions.jsonl','duplicate'),('predictions.jsonl','unbound')])
def test_receipt_hashes_do_not_substitute_for_unique_matching_row_coverage(tmp_path,artifact,mode):
    records=validate_records([record(i) for i in range(3)],tmp_path)
    run=run_api(records,tmp_path,tmp_path/'run',FakeProvider('openai',lambda t:'yes')).parent
    path=run/artifact
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    if mode=='missing': rows.pop()
    elif mode=='duplicate': rows[-1]=dict(rows[0])
    else:
        # Replace a selected prediction with another valid dataset ID, not a raw-bound ID.
        extra=validate_records([record(9)],tmp_path)[0]
        records.append(extra)
        rows[-1]['id']=extra['id']
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    completion=json.loads((run/'completion.json').read_text())
    completion['raw_sha256' if artifact=='raw.jsonl' else 'predictions_sha256']=file_digest(path)
    (run/'completion.json').write_text(json.dumps(completion))
    with pytest.raises(ValueError,match='coverage|unique|count'):
        prelabel_export(run,records)

def test_subset_run_can_still_be_imported_into_larger_dataset(tmp_path):
    records=validate_records([record(i) for i in range(3)],tmp_path)
    run=run_api(records[:2],tmp_path,tmp_path/'run',FakeProvider('openai',lambda t:'yes')).parent
    exported=prelabel_export(run,records)
    assert [x['id'] for x in exported['labels']]==['r0','r1']
    assert exported['errors']==0

def test_completion_count_must_match_manifest(tmp_path):
    records=validate_records([record(1)],tmp_path)
    run=run_api(records,tmp_path,tmp_path/'run',FakeProvider('openai',lambda t:'yes')).parent
    completion=json.loads((run/'completion.json').read_text()); completion['completed_count']=999
    (run/'completion.json').write_text(json.dumps(completion))
    with pytest.raises(ValueError,match='count'): prelabel_export(run,records)
