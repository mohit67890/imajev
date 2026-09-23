from imajev_bench.annotations import merge_reviews
from imajev_bench.runner import digest
from imajev_bench.schema import model_payload


def row(id):
    return {"id":id,"group_id":"shared","track":"text","family":"rule","split":"dev","images":[],
            "request":{"request_id":id,"state":"value: 3","fields":[{"id":"q","type":"boolean","question":"Is it 3?"}]},
            "gold":True,"annotation_status":"draft","provenance":{}}


def test_flag_quarantines_group_even_with_two_agreements(tmp_path):
    a,b=row('a'),row('b')
    reviews=[{"id":"a","reviewer_id":r,"value":True,"evidence":"Three visible","input_sha256":digest(model_payload(a))} for r in ('alice','bob')]
    flags=[{"id":"a","reviewer_id":"alice","reason":"Rule is ambiguous","input_sha256":digest(model_payload(a))}]
    out=merge_reviews([a,b],[{"format_version":"0.0.1","purpose":"independent_review","reviews":reviews,"flags":flags}],tmp_path)
    assert all(r['annotation_status']=='draft' and r['provenance']['quarantined'] for r in out)
    assert out[0]['provenance']['review_flags'][0]['reason']=='Rule is ambiguous'
