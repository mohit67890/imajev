import json
import pytest
from imajev_bench.api_models import GeminiProvider,parse_reply

class Reply:
    def __init__(self,parts): self.parts=parts
    def read(self): return json.dumps({'modelVersion':'fixture','candidates':[{'content':{'parts':self.parts}}]}).encode()
    def __enter__(self): return self
    def __exit__(self,*args): pass

@pytest.mark.parametrize('parts,expected',[
 ([{'thought':True,'text':'A discarded possibility is {"answer":"no"}.'},{'text':'{"answer":"yes","evidence":"state"}'}],'answered'),
 ([{'thought':True,'text':'{"answer":"yes","evidence":"only a thought"}'}],'error'),
 ([{'text':'{"answer":"yes",'},{'text':'"evidence":"state"}'}],'answered'),
 ([{'thought':False,'text':'{"answer":"no","evidence":"state"}'}],'answered'),
])
def test_only_final_text_parts_are_decoded(parts,expected):
    provider=GeminiProvider('fixture','not-a-key',opener=lambda *a,**k:Reply(parts))
    text,model,usage=provider.complete('question',[],['yes','no'],True)
    result=parse_reply(text,[('yes',True,None),('no',False,None)])
    assert result['status']==expected
    assert model=='fixture'
    if expected=='answered': assert 'only a thought' not in text and 'discarded possibility' not in text
