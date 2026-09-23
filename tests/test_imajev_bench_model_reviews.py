import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
from imajev_bench.annotations import merge_reviews
from imajev_bench.runner import digest
from imajev_bench.schema import model_payload,validate_records
spec=importlib.util.spec_from_file_location('comparison',Path(__file__).parents[1]/'scripts/imajev_bench/compare_model_reviews.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def fixture(tmp_path):
    rows=validate_records([{'id':'a','group_id':'g','track':'text','family':'f','split':'dev','images':[],'request':{'request_id':'a','state':'yes','fields':[{'id':'q','type':'boolean','question':'Yes?'}]},'gold':True,'annotation_status':'draft','provenance':{}}],tmp_path)
    sha=digest(model_payload(rows[0]));ds=hashlib.sha256(json.dumps([['a',sha]],separators=(',',':')).encode()).hexdigest()
    def ex(name,value):return {'format_version':'0.0.1','protocol':'imajev-bench-blind-review-v1','purpose':'independent_review','annotator_type':'model','reviewer_id':name,'input_sha256':ds,'reviews':[{'id':'a','reviewer_id':name,'input_sha256':sha,'value':value,'evidence':'yes'}],'flags':[]}
    return rows,[ex('a',True),ex('b',True)]

def test_consensus_and_disagreement(tmp_path):
    rows,exports=fixture(tmp_path)
    assert module.compare(rows,exports,tmp_path)['consensus_count']==1
    exports[1]['reviews'][0]['value']=False
    result=module.compare(rows,exports,tmp_path)
    assert result['consensus_count']==0 and result['groups_requiring_review']==['g']
    assert rows[0]['annotation_status']=='draft'

def test_identity_hash_and_completeness(tmp_path):
    rows,exports=fixture(tmp_path)
    for mutate in [lambda e:e.update(input_sha256='bad'),lambda e:e.update(reviews=[]),lambda e:e.update(reviewer_id='a')]:
        altered=copy.deepcopy(exports);mutate(altered[1])
        with pytest.raises(ValueError):module.compare(rows,altered,tmp_path)

def test_model_cannot_promote_human_review(tmp_path):
    rows,exports=fixture(tmp_path)
    with pytest.raises(ValueError,match='separate'):merge_reviews(rows,exports,tmp_path)
