import json
import pytest
from imajev_bench.cli import main
from imajev_bench.runner import run,file_digest
from imajev_bench.schema import validate_records
from test_imajev_bench_triage import record

@pytest.mark.parametrize('badid',['duplicate','outside',None,42])
def test_compare_rejects_duplicate_and_out_of_split_prediction_ids(tmp_path,badid):
    records=validate_records([record(1),record(2)],tmp_path)
    records_file=tmp_path/'records.jsonl'; records_file.write_text(''.join(json.dumps(r)+'\n' for r in records))
    a=run(records,tmp_path,tmp_path/'a'); b=run(records,tmp_path,tmp_path/'b')
    rows=[json.loads(x) for x in b.read_text().splitlines()]
    rows[-1]['id']=rows[0]['id'] if badid=='duplicate' else badid
    b.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    completion=json.loads((b.parent/'completion.json').read_text()); completion['predictions_sha256']=file_digest(b)
    (b.parent/'completion.json').write_text(json.dumps(completion))
    output=tmp_path/'comparison.json'
    with pytest.raises(ValueError,match='Prediction|Predictions'):
        main(['compare','--records',str(records_file),'--allow-draft','--split','dev','--predictions-a',str(a),'--predictions-b',str(b),'--output',str(output)])
    assert not output.exists()

def test_compare_control_keeps_complete_runs_and_paired_result(tmp_path):
    records=validate_records([record(1),record(2)],tmp_path)
    records_file=tmp_path/'records.jsonl'; records_file.write_text(''.join(json.dumps(r)+'\n' for r in records))
    a=run(records,tmp_path,tmp_path/'a'); b=run(records,tmp_path,tmp_path/'b')
    output=tmp_path/'comparison.json'
    main(['compare','--records',str(records_file),'--allow-draft','--split','dev','--predictions-a',str(a),'--predictions-b',str(b),'--output',str(output)])
    result=json.loads(output.read_text()); assert result['records']==2 and result['difference']==0 and result['p_value']==1
