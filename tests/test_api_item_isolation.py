import json
from imajev_bench.api_models import run_api,prelabel_export
from imajev_bench.runner import verify_run
from imajev_bench.schema import validate_records

class Provider:
    name='openai'; model='fixture'
    def __init__(self): self.calls=[]
    def complete(self,text,images,tokens,constrained):
        self.calls.append(tokens)
        return json.dumps({'answer':tokens[0],'evidence':'fixture'}),'fixture',{}

def record(name,options):
    return {'id':name,'group_id':name,'track':'text','family':'choice','split':'dev','images':[],
      'request':{'request_id':name,'state':{},'fields':[{'id':'q','type':'choice','question':'Choose','options':[{'value':x} for x in options]}]},
      'gold':options[0],'annotation_status':'draft','provenance':{}}

def test_unsupported_item_is_an_error_and_valid_following_items_complete(tmp_path):
    rows=validate_records([record('bad',['unknown','other']),record('good',['a','b'])],tmp_path)
    p=Provider()
    path=run_api(rows,tmp_path,tmp_path/'run',p,workers=2,order_seed=None)
    predictions=[json.loads(x) for x in path.read_text().splitlines()]
    assert [x['id'] for x in predictions]==['bad','good']
    assert predictions[0]['status']=='error' and predictions[0]['error_type']=='ValueError'
    assert predictions[1]['status']=='answered' and predictions[1]['value']=='a'
    assert p.calls==[['a','b']]
    verify_run(rows,path)
    exported=prelabel_export(path.parent,rows)
    assert exported['errors']==1 and [x['id'] for x in exported['labels']]==['good']

def test_ordinary_items_keep_prompt_hash_and_options(tmp_path):
    rows=validate_records([record('good',['a','b'])],tmp_path)
    path=run_api(rows,tmp_path,tmp_path/'run',Provider(),order_seed=None)
    raw=json.loads((path.parent/'raw.jsonl').read_text())
    assert raw['option_order']==['a','b'] and len(raw['prompt_sha256'])==64
