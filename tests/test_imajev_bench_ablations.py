import copy,hashlib,importlib.util,json
from pathlib import Path
import pytest


def load(name):
    p=Path(__file__).parents[1]/'scripts/imajev_bench'/f'{name}.py';s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
builder=load('build_ablations');analysis=load('analyze_ablations')


def setup(tmp_path):
    (tmp_path/'image.jpg').write_bytes(b'asset');sha=hashlib.sha256(b'asset').hexdigest()
    row={'id':'source','group_id':'scene','track':'joint','family':'written_rule','split':'dev','images':[{'path':'image.jpg','sha256':sha}],
         'request':{'request_id':'source','state':{'threshold':3,'policy':'below threshold'},'fields':[{'id':'q','type':'boolean','question':'Accept?'}]},'gold':True,'annotation_status':'draft','provenance':{'draft_evidence':'secret label'}}
    manifest=builder.build([row],tmp_path,tmp_path/'out');packets={k:json.loads(Path(v['packet']).read_text()) for k,v in manifest['conditions'].items()}
    exports={k:{'protocol':'imajev-modality-probe-v1','predictions':[{'id':p['items'][0]['id'],'input_sha256':p['items'][0]['input_sha256'],'value':True if k=='full' else None,'evidence':'test'}]} for k,p in packets.items()}
    return manifest,packets,exports


def test_only_intended_inputs_change_and_gold_never_reaches_packet(tmp_path):
    m,p,e=setup(tmp_path);f=p['full']['items'][0];i=p['no_image']['items'][0];s=p['no_state']['items'][0]
    assert f['request']==i['request'] and not i['images'] and f['images']
    assert s['images']==f['images'] and s['request']['fields']==f['request']['fields'] and s['request']['state']=={}
    assert len({x['items'][0]['input_sha256'] for x in p.values()})==3
    assert 'full_reference' not in json.dumps(p) and 'secret label' not in json.dumps(p)


def test_unknown_transition_not_called_ablation_accuracy(tmp_path):
    m,p,e=setup(tmp_path);r=analysis.summarize(m,p,e)
    assert r['paired']['no_image']['full_correct_to_unknown']==1
    assert r['conditions']['no_image']['strata']['all']['full_reference_matches']==0
    assert 'accuracy' not in r['conditions']['no_image']['strata']['all']


def test_hash_identity_and_type_guards(tmp_path):
    m,p,e=setup(tmp_path)
    for update in [{'input_sha256':'bad'},{'value':1},{'id':'wrong'}]:
        x=copy.deepcopy(e['full']);x['predictions'][0].update(update)
        with pytest.raises(ValueError):analysis.validate_predictions(p['full'],x)
    x=copy.deepcopy(e['full']);x['predictions']*=2
    with pytest.raises(ValueError):analysis.validate_predictions(p['full'],x)
