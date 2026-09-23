import hashlib, json, sys, zipfile
from pathlib import Path
import pandas as pd

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from v1_text.convert_claims_reading import candidate_spans, dedupe_snli, fever_rows, phrase
from vision_decision.contracts import Request, UNKNOWN

def license(root):
 p=root/'LICENSE.txt';p.write_text('terms');(root/'LICENSE.txt.receipt.json').write_text(json.dumps({'spdx':'CC-BY-SA-3.0','evidence_sha256':hashlib.sha256(b'terms').hexdigest(),'commercial_use_reviewed':True}));return p

def test_fever_resolves_sentences_without_leaking_label(tmp_path,monkeypatch):
 raw=tmp_path/'raw';raw.mkdir();lic=tmp_path/'lic';lic.mkdir();evidence=license(lic)
 support={'id':1,'label':'SUPPORTS','claim':'Sky is blue.','evidence':[[[0,0,'Sky',1]]]};nei={'id':2,'label':'NOT ENOUGH INFO','claim':'Moon is cheese.','evidence':[[[0,0,None,None]]]}
 for name,rows in [('train.jsonl',[support,nei]),('paper_dev.jsonl',[support]),('paper_test.jsonl',[nei])]:Path(raw/name).write_text(''.join(json.dumps(x)+'\n' for x in rows))
 with zipfile.ZipFile(raw/'wiki-pages.zip','w') as z:z.writestr('wiki-pages/wiki-001.jsonl',json.dumps({'id':'Sky','lines':'0\tClouds exist\n1\tSky is blue'})+'\n')
 monkeypatch.setattr('v1_text.convert_claims_reading.LIC',lic.parent)
 # The converter expects LIC/fever; point verified evidence there.
 (lic.parent/'fever').mkdir();(lic.parent/'fever'/'LICENSE.txt').write_bytes(evidence.read_bytes());(lic.parent/'fever'/'LICENSE.txt.receipt.json').write_text((lic/'LICENSE.txt.receipt.json').read_text())
 rows=fever_rows(raw,10,10);answered=next(x for x in rows if x['target']=='supported');unknown=next(x for x in rows if x['target'] is None)
 assert answered['request']['state']=={'claim':'Sky is blue.','evidence':['Sky is blue']}
 assert 'label' not in json.dumps(answered['request']).lower() and unknown['abstention_cause']=='insufficient_evidence'
 for row in rows:Request.model_validate(row['request'])

def test_candidate_spans_are_deterministic_and_include_gold():
 a=candidate_spans('Paris is in France and Berlin is in Germany.','Where is Paris?','France','x')
 assert a==candidate_spans('Paris is in France and Berlin is in Germany.','Where is Paris?','France','x') and a[0]=='France' and len(a)>=2

def test_every_source_has_six_deterministic_phrasings():
 for source in ('fever','boolq','snli','squad2'):
  assert len({phrase(source,str(i)) for i in range(100)})==6

def test_snli_collapses_duplicates_and_rejects_conflicting_gold():
 rows=[{'premise':'p','hypothesis':'h','label':0},{'premise':'p','hypothesis':'h','label':0},
       {'premise':'x','hypothesis':'y','label':1},{'premise':'x','hypothesis':'y','label':2}]
 assert dedupe_snli(rows)==[rows[0]]
