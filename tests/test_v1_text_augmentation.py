from copy import deepcopy
from v1_text.augment_text import augment, reference_controls
from v1_text.common import write_jsonl, read_rows
from decision_data import expand_fields, render


def row(i=0):
 return {'id':f'fixture:{i}','source':'abo','source_group':'group','partition':'train','images':[],
         'request':{'request_id':f'fixture-{i}','state':{'listing':{'brand':'A'}},'fields':[{'id':'brand','type':'choice','question':'Which brand is in the listing?','options':[{'value':x} for x in ['A','B','C']]}]},'target':'A','abstention_cause':None}


def test_abstention_controls_keep_group_and_render_validly():
 base=[row(i) for i in range(100)]
 controls=augment(base)+reference_controls(base,10)
 assert len(controls)>len(base)
 for r in controls:
  assert r['source_group']=='group'
  for item in expand_fields(r):render(item)
  if r.get('abstention_cause')=='not_listed':assert all(o['value']!='A' for o in r['request']['fields'][0]['options'])
 assert {r.get('abstention_cause') for r in controls}>={'not_listed','insufficient_evidence','false_premise','mismatched_reference'}
 assert base==[row(i) for i in range(100)]


def test_unicode_jsonl_and_eval_unchanged(tmp_path):
 r=row();r['partition']='test';r['request']['state']='a\u2028b\u0085c'
 assert augment([r])==[r]
 p=tmp_path/'records.jsonl';write_jsonl(p,[r]);assert read_rows(p)==[r]
 assert len(p.read_text().splitlines())==1
