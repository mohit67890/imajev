import json,sys
sys.path.insert(0,'scripts')
from v1_text.convert_routing_emotion import choice_record,group,select

LIC={'spdx':'MIT','evidence':'x','evidence_sha256':'x','receipt':'x'}
def test_oos_is_unknown_and_options_are_deterministically_permuted():
 a=choice_record('clinc150','test','odd request','oos',['a','b','c'],LIC,'1',oos=True)
 b=choice_record('clinc150','test','odd request','oos',['c','a','b'],LIC,'1',oos=True)
 assert a['target'] is None and a['abstention_cause']=='not_listed' and a['request']==b['request']
def test_group_normalizes_content_and_namespaces_source():
 assert group('a',' Hello   WORLD ')==group('a','hello world') and group('a','hello')!=group('b','hello')
def test_selection_preserves_eval_precedence_and_caps_decisions():
 def row(i,p):return {'id':str(i),'source':'banking77','source_group':'g' if i<2 else f'g{i}','partition':p,'target':1}
 rows=select([row(0,'train'),row(1,'test')]+[row(i,'train') for i in range(2,13010)])
 assert not any(r['id']=='0' for r in rows) and any(r['id']=='1' for r in rows)
 assert sum(r['partition']=='train' for r in rows)==12000
