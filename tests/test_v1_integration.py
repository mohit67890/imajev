import json, sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from v1_text.assemble_mixture import stratified_train
from v1_text.augment_text import augment, rank
from v1_text.build_controls import add_controls
from v1_text.deduplicate_text import deduplicate

def row(rid,state='evidence',target='yes',fields=None,partition='train'):
 return {'id':rid,'source':'s','family':'f','source_group':rid,'partition':partition,'images':[],
  'request':{'schema_version':'1.0','request_id':rid,'state':state,'fields':fields or [{'id':'q','type':'choice','question':'Choose.','options':[{'value':'yes'},{'value':'no'}]}]},
  'target':target,'abstention_cause':None}

def control_id(cause_parity=None):
 for i in range(10000):
  r=row(f'r{i}')
  if rank(r,'abstain')%100<12 and (cause_parity is None or rank(r,'cause')%2==cause_parity):return r['id']
 raise AssertionError('no deterministic fixture id')

def test_unknown_augmentation_requires_removed_evidence_and_no_literal_answer():
 rid=control_id(0) # missing-evidence branch rather than not-listed
 assert len(augment([row(rid,state={})]))==1
 leaking=row(rid,state={'passage':'context'});leaking['request']['fields'][0]['question']='Is the answer yes?'
 assert len(augment([leaking]))==1
 safe=row(rid,state={'passage':'context'});made=augment([safe]);assert len(made)==2
 assert made[1]['target'] is None and made[1]['request']['state']=={} and made[1]['parent_id']==rid

def test_training_state_presentation_is_about_40_percent_string_and_lossless():
 original=[row(f'mix-{i}',state={'ticket':{'id':i,'status':'open'}}) for i in range(1000)]
 made=[r for r in augment(original) if not r.get('parent_id')]
 strings=[r for r in made if isinstance(r['request']['state'],str)]
 assert 350<=len(strings)<=450
 for r in made:
  state=r['request']['state'];state=json.loads(state) if isinstance(state,str) else state
  assert state['ticket']['status']=='open'

def test_short_exact_state_obeys_heldout_precedence():
 train=row('train',state='same',partition='train');test=row('test',state='same',partition='test')
 kept,removed=deduplicate([train,test]);assert kept==[test] and removed=={'s/train':1}

def test_expanded_fields_count_toward_limit():
 three=row('three',fields=[{'id':f'q{i}','type':'boolean','question':'Q?'} for i in range(3)])
 two=row('two',fields=[{'id':f'q{i}','type':'boolean','question':'Q?'} for i in range(2)])
 chosen=stratified_train([three,two],4,'seed')
 assert sum(len(r['request']['fields']) for r in chosen)<=4

def test_irrelevant_image_donor_must_be_train_exclusive():
 shared={'id':'im-train','source':'im','family':'im','source_group':'g','partition':'train','images':[{'image':'x','sha256':'shared'}],'request':{'fields':[{}]}}
 shared_test={**json.loads(json.dumps(shared)),'id':'im-test','partition':'test'}
 with pytest.raises(ValueError,match='require training images'):
  add_controls([row('text')],[shared,shared_test],text_image_rate=1)
 exclusive={**json.loads(json.dumps(shared)),'id':'exclusive','images':[{'image':'y','sha256':'exclusive'}]}
 made=add_controls([row('text')],[shared,shared_test,exclusive],text_image_rate=1)
 attached=next(r for r in made if r['id']=='text')
 assert attached['images'][0]['sha256']=='exclusive' and attached['partition']=='train'
