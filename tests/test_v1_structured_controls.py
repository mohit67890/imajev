import json,sys
from pathlib import Path
sys.path.insert(0,'scripts')
from v1_text.convert_structured import LEVELS,abo_image_groups,keyhash,record
from v1_text.build_controls import grounded_contradictions
from decision_data import expand_fields

def test_esci_levels_have_correct_ordinal_semantics():
    assert [x['value'] for x in LEVELS]==[0,1,2,3]
    assert ['Irrelevant','Complement','Substitute','Exact']==[x['description'].split(':',1)[0] for x in LEVELS]

def test_abo_image_mapping_rejects_ambiguous_item_and_group(tmp_path):
    rows=[
      {'source':'abo','id':'abo:B000000001-a','source_group':'g1','partition':'train'},
      {'source':'abo','id':'abo:B000000001-b','source_group':'g2','partition':'train'},
      {'source':'abo','id':'abo:B000000002-a','source_group':'g3','partition':'train'},
      {'source':'abo','id':'abo:B000000003-a','source_group':'g4','partition':'train'},
      {'source':'abo','id':'abo:B000000004-a','source_group':'g4','partition':'test'},
      {'source':'other','id':'other:B000000005','source_group':'x','partition':'test'}]
    p=tmp_path/'manifest.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    mapping,ambiguous=abo_image_groups(p)
    assert mapping=={'B000000002':('g3','train')}
    assert {'B000000001','B000000003','B000000004'}<=ambiguous

def test_record_keeps_source_group_and_target_exactly():
    field={'id':'relevance','type':'ordinal','question':'Relevant?','levels':LEVELS}
    r=record('esci','1',keyhash('query'),'train','calibration',{'query':'q'},field,3,{'spdx':'x'})
    assert r['partition']=='calibration' and r['target']==3 and r['source_group']==keyhash('query')

def test_grounded_contradiction_keeps_photo_and_listing_answers_separate():
    def donor(ident,category,sha):
        return {'id':ident,'source':'abo','source_group':ident,'source_answer':'category='+category,
                'partition':'train','family':'abo','abstention_cause':None,
                'images':[{'image':'unused.jpg','sha256':sha}],
                'request':{'schema_version':'1.0','request_id':ident,'state':{},
                           'fields':[{'id':'old','type':'boolean','question':'old'}]},'target':True}
    rows=grounded_contradictions([donor('a','shoe','1'),donor('b','chair','2')],limit=2)
    assert len(rows)==2
    for row in rows:
        assert row['partition']=='train' and row['source_group'] in {'a','b'}
        assert row['targets']['photo']!=row['targets']['listing']
        expanded={r['request']['fields'][0]['id']:r['target'] for r in expand_fields(row)}
        assert expanded==row['targets']
        assert {o['value'] for o in row['request']['fields'][0]['options']}==set(row['targets'].values())
