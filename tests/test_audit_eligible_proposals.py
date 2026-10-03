import pytest
from imajev_bench.triage import audit_report
from imajev_bench.runner import digest
from imajev_bench.schema import model_payload,validate_records
from test_imajev_bench_triage import record,human_export
from imajev_bench.annotations import merge_reviews

def reviewed(tmp_path):
    records=validate_records([record(1),record(2)],tmp_path)
    records=merge_reviews(records,[human_export(records,'human-a',{r['id']:True for r in records}),human_export(records,'human-b',{r['id']:True for r in records})],tmp_path)
    for row in records:
        row['provenance']['audit_sample']=True
        row['provenance']['model_prelabels']=[{'provider':p,'value':True,'input_sha256':digest(model_payload(row))} for p in ('a','b')]
    return records

def test_stale_proposal_does_not_dilute_audited_consensus_errors(tmp_path):
    rows=reviewed(tmp_path)
    rows[0]['provenance']['model_prelabels'][0]['input_sha256']='0'*64
    rows[1]['provenance']['model_prelabels'][0]['value']=False
    rows[1]['provenance']['model_prelabels'][1]['value']=False
    result=audit_report(rows)
    assert result['audited']==1 and result['consensus_errors']==1 and result['error_rate']==1
    assert result['excluded_from_error_rate']==['r1']

def test_no_current_consensus_yields_no_audited_error_estimate(tmp_path):
    rows=reviewed(tmp_path)
    for row in rows: row['provenance'].pop('model_prelabels')
    result=audit_report(rows)
    assert result['audited']==0 and result['error_rate'] is None and result['wilson_upper_95'] is None
    assert result['excluded_from_error_rate']==['r1','r2']

def test_valid_constructed_and_current_consensus_audits_remain_counted(tmp_path):
    rows=reviewed(tmp_path)
    rows[0]['provenance']['construction']={'truth_source':'programmatic','truth':True}
    result=audit_report(rows)
    assert result['audited']==2 and result['error_rate']==0 and result['excluded_from_error_rate']==[]

@pytest.mark.parametrize('verdict',['corrected','excluded'])
def test_known_model_audit_errors_are_retained_without_current_consensus(tmp_path,verdict):
    rows=reviewed(tmp_path)
    for row in rows:
        row['provenance'].pop('model_prelabels')
        row['provenance']['model_audit']={'verdict':verdict}
    result=audit_report(rows)
    assert result['audited']==2 and result['consensus_errors']==2 and result['error_rate']==1
    assert result['excluded_from_error_rate']==[] and result['audit_method']=='model'
