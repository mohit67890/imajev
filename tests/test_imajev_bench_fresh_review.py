import hashlib
import json

import pytest

from imajev_bench.annotations import merge_reviews
from imajev_bench.audit import audit
from imajev_bench.review import build_review
from imajev_bench.runner import digest
from imajev_bench.schema import model_payload


def record(rid, track='text'):
    return {'id':rid,'group_id':'scene','track':track,'family':'rule','split':'dev','images':[],
            'request':{'request_id':rid,'state':'Three units','fields':[{'id':'q','type':'boolean','question':'Three?'}]},
            'gold':True,'annotation_status':'draft','provenance':{}}


def test_mixed_track_scene_counted_in_both_tracks():
    result=audit([record('a','visual'),record('b','joint')])
    assert result['declared_groups']==1
    assert result['track_groups']=={'visual':1,'joint':1}
    assert result['mixed_track_groups']==1


def test_export_bound_to_dataset_and_declared_reviewer(tmp_path):
    row=record('a')
    sha=digest(model_payload(row))
    export={'format_version':'0.0.1','protocol':'imajev-bench-blind-review-v1','purpose':'independent_review',
            'input_sha256':hashlib.sha256(json.dumps([['a',sha]],separators=(',',':')).encode()).hexdigest(),
            'reviewer_id':'alice','reviews':[{'id':'a','reviewer_id':'alice','input_sha256':sha,'value':True,'evidence':'Three'}]}
    assert merge_reviews([row],[export],tmp_path)[0]['annotation_status']=='draft'
    export['reviewer_id']='bob'
    with pytest.raises(ValueError,match='declared reviewer'):
        merge_reviews([row],[export],tmp_path)
    export['reviewer_id']='alice';export['input_sha256']='0'*64
    with pytest.raises(ValueError,match='dataset input hash'):
        merge_reviews([row],[export],tmp_path)


def test_photo_credits_are_preserved_and_escaped(tmp_path):
    image=tmp_path/'a.jpg';image.write_bytes(b'asset')
    row=record('a','visual');row['images']=[{'path':'a.jpg','sha256':hashlib.sha256(b'asset').hexdigest()}]
    row['provenance']={'creator':'<script>bad</script>','license':'CC BY 4.0','license_url':'https://creativecommons.org/licenses/by/4.0/','source_page':'https://commons.wikimedia.org/wiki/File:Example.jpg'}
    result=build_review([row],tmp_path,tmp_path/'review.html').read_text()
    assert 'CC BY 4.0' in result and 'consult after independent review' in result
    assert '<script>bad</script>' not in result
    assert '&lt;script&gt;bad&lt;/script&gt;' in result
