import copy
from imajev_bench.release import release_check
from imajev_bench.schema import validate_records
from test_imajev_bench_triage import record

def reviewed(tmp_path,i,gold=True):
    row=record(i,gold=gold,split='test',review_route='construction_verified',construction={'truth_source':'programmatic','truth':gold})
    row['annotation_status']='reviewed'
    return validate_records([row],tmp_path)[0]

def test_quarantined_rows_cannot_inflate_scored_release_gates(tmp_path):
    eligible=reviewed(tmp_path,1)
    quarantined=record(2,gold=None,split='test',quarantined=True,judgement_dependent=True)
    quarantined=validate_records([quarantined],tmp_path)[0]
    reference=release_check([eligible],tmp_path)
    actual=release_check([eligible,quarantined],tmp_path)
    before={g['gate']:g for g in reference['gates']}; after={g['gate']:g for g in actual['gates']}
    for name in before:
        if name!='human_review': assert after[name]==before[name],name
    assert after['independent_clusters']['counts']['test/text']==1
    assert '1 quarantined' in after['human_review']['detail']

def test_quarantine_does_not_affect_annotator_agreement(tmp_path):
    eligible=reviewed(tmp_path,1)
    bad=record(2,split='test',quarantined=True,review_plan='double')
    bad['provenance']['reviews']=[{'value':True},{'value':False}]
    bad=validate_records([bad],tmp_path)[0]
    gates={g['gate']:g for g in release_check([eligible,bad],tmp_path)['gates']}
    assert gates['annotator_agreement']['passed'] and gates['annotator_agreement']['pairs']==0

def test_all_quarantined_dataset_is_explicitly_not_ready(tmp_path):
    row=validate_records([record(1,quarantined=True)],tmp_path)[0]
    report=release_check([row],tmp_path)
    assert report['release_ready'] is False
    gates={g['gate']:g for g in report['gates']}
    assert not gates['human_review']['passed']
    assert not gates['construction_lint']['passed']
    assert all(value==0 for value in gates['independent_clusters']['counts'].values())
