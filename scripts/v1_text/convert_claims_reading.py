"""Convert approved FEVER, BoolQ, SNLI and SQuAD 2.0 releases into decision records."""
from __future__ import annotations
import argparse, collections, hashlib, json, re, unicodedata, zipfile
from pathlib import Path
from .common import fit_partition, read_rows, verified_license, write_jsonl

ROOT=Path(__file__).resolve().parents[2]; RAW=ROOT/'data/decision-v1-text/raw'; LIC=ROOT/'data/decision-v1-text/licenses'
OUT=ROOT/'data/decision-v1-text/converted/claims-reading.jsonl'; REPORT=ROOT/'reports/v1.1-datasets/claims-reading-conversion.json'
PHRASINGS={
 'fever':['Classify this claim from the supplied evidence.','Does the evidence support, refute, or fail to determine the claim?','Judge the claim using only the evidence.','Select the evidence-grounded verdict.','What follows from the provided evidence?','Determine the claim status without outside knowledge.'],
 'boolq':['Answer the question from the passage.','Using only the passage, is the answer yes or no?','Decide whether the passage supports a yes answer.','Read the passage and answer the yes/no question.','Is the question true according to the passage?','Return the passage-grounded answer.'],
 'snli':['Classify the relationship between the premise and hypothesis.','Does the premise entail, contradict, or leave the hypothesis neutral?','Judge the hypothesis using only the premise.','Select the natural-language inference relation.','What relation holds between these two statements?','Determine whether the hypothesis follows from the premise.'],
 'squad2':['Select the answer span from the passage.','Which candidate answers the question from the passage?','Choose the passage-grounded answer.','Identify the best answer span, or unknown if unanswered.','Answer using one listed span from the passage.','Which candidate is supported by the context?']}

def rank(seed,*parts):return hashlib.sha256(('\0'.join((seed,*map(str,parts)))).encode()).hexdigest()
def phrase(source,key):return PHRASINGS[source][int(rank('phrasing',key)[:8],16)%len(PHRASINGS[source])]
def group_hash(prefix,text):return prefix+':'+hashlib.sha256(text.encode()).hexdigest()[:20]
def select(rows,n,key=lambda r:r.get('id',''),strata=None):
 if strata:
  buckets=collections.defaultdict(list)
  for r in rows:buckets[strata(r)].append(r)
  out=[];base=n//len(buckets);extra=n%len(buckets)
  for i,k in enumerate(sorted(buckets)):out+=sorted(buckets[k],key=lambda r:rank('select',key(r)))[:base+(i<extra)]
  return sorted(out,key=lambda r:rank('merge',key(r)))
 return sorted(rows,key=lambda r:rank('select',key(r)))[:n]
def options(values,key):
 ordered=sorted(values,key=lambda x:rank('options',key,x[0]));return [{'value':v,**({'description':d} if d else {})} for v,d in ordered]
def dedupe_snli(rows):
 pairs=collections.defaultdict(list)
 for r in rows:pairs[r['premise']+'\0'+r['hypothesis']].append(r)
 return [values[0] for values in pairs.values() if len({int(x['label']) for x in values})==1]
def record(source,rid,group,source_split,partition,state,question,field,target,license_info,cause=None):
 return {'id':f'{source}:{rid}','source':source,'source_split':source_split,'source_group':group,'family':source,
  'heldout_family':partition=='test','license':license_info,'images':[],'request':{'schema_version':'1.0','request_id':f'{source}-{rid}','state':state,
  'fields':[{'id':'decision','question':question,**field}]},'target':target,'abstention_cause':cause,'partition':partition}

def fever_rows(raw=RAW/'fever',limit_train=20000,limit_eval=1000):
 lic=verified_license(LIC/'fever/LICENSE.txt','CC-BY-SA-3.0');train=read_rows(raw/'train.jsonl');dev=read_rows(raw/'paper_dev.jsonl');test=read_rows(raw/'paper_test.jsonl')
 train=select(train,min(limit_train,len(train)),lambda r:r['id'],lambda r:r['label']);dev=select(dev,min(limit_eval,len(dev)),lambda r:r['id'],lambda r:r['label']);test=select(test,min(limit_eval,len(test)),lambda r:r['id'],lambda r:r['label']);chosen=train+dev+test
 needed={e[2] for r in chosen for group in r.get('evidence',[]) for e in group if len(e)>=4 and e[2] is not None};by_normal={unicodedata.normalize('NFC',x):x for x in needed};pages={}
 with zipfile.ZipFile(raw/'wiki-pages.zip') as z:
  for info in z.infolist():
   if not info.filename.startswith('wiki-pages/wiki-') or not info.filename.endswith('.jsonl'):continue
   for line in z.open(info):
    if not line.strip():continue
    row=json.loads(line)
    original=by_normal.get(unicodedata.normalize('NFC',row['id']))
    if original is not None:pages[original]=row['lines']
   if len(pages)==len(needed):break
 def evidence(r):
  found=[]
  for evidence_set in r.get('evidence',[]):
   for e in evidence_set:
    if len(e)<4 or e[2] is None:continue
    lines=pages.get(e[2],'').splitlines();prefix=str(e[3])+'\t'
    match=next((x[len(prefix):] for x in lines if x.startswith(prefix)),None)
    if match and match not in found:found.append(match)
  return found
 out=[]
 for split,rows_ in [('train',train),('dev',dev),('test',test)]:
  for r in rows_:
   rid=str(r['id']);label=r['label'];ev=evidence(r);target={'SUPPORTS':'supported','REFUTES':'refuted','NOT ENOUGH INFO':None}[label]
   if target is not None and not ev:raise ValueError(f'FEVER {rid}: cited evidence sentence missing')
   partition=fit_partition(rid) if split=='train' else split
   state={'claim':r['claim'],'evidence':ev} # no gold label or annotation IDs
   field={'type':'choice','options':options([('supported','the evidence supports the claim'),('refuted','the evidence contradicts the claim')],rid)}
   out.append(record('fever',rid,rid,split,partition,state,phrase('fever',rid),field,target,lic,'insufficient_evidence' if target is None else None))
 return out

def boolq_rows(raw=RAW/'boolq',limit_eval=2000):
 lic=verified_license(LIC/'boolq/LICENSE.txt','CC-BY-SA-3.0');out=[]
 validation=read_rows(raw/'validation.parquet');heldout_groups={group_hash('passage',r['passage']) for r in validation}
 for split,path in [('train',raw/'train.parquet'),('validation',raw/'validation.parquet')]:
  rows=read_rows(path);rows=[r for r in rows if split!='train' or group_hash('passage',r['passage']) not in heldout_groups];rows=rows if split=='train' else select(rows,min(limit_eval,len(rows)),lambda r:r['question']+r['passage'])
  for r in rows:
   group=group_hash('passage',r['passage']);rid=rank('boolq',r['question'],r['passage'])[:20];partition=fit_partition(group) if split=='train' else 'test'
   q=phrase('boolq',rid)+'\nQuestion: '+r['question'];field={'type':'boolean','yes_description':'yes according to the passage','no_description':'no according to the passage'}
   out.append(record('boolq',rid,group,split,partition,r['passage'],q,field,bool(r['answer']),lic))
 return out

def snli_rows(raw=RAW/'snli',limit_train=20000,limit_dev=1000,limit_test=2000):
 lic=verified_license(LIC/'snli/LICENSE.txt','CC-BY-SA-4.0');mapping={0:'entailment',1:'neutral',2:'contradiction'};out=[]
 all_rows={s:[r for r in read_rows(raw/f'{s}.parquet') if int(r['label']) in mapping] for s in ('train','validation','test')}
 test_groups={group_hash('premise',r['premise']) for r in all_rows['test']};dev_groups={group_hash('premise',r['premise']) for r in all_rows['validation']}-test_groups
 for split,limit in [('train',limit_train),('validation',limit_dev),('test',limit_test)]:
  rows=all_rows[split]
  if split=='train':rows=[r for r in rows if group_hash('premise',r['premise']) not in test_groups|dev_groups]
  elif split=='validation':rows=[r for r in rows if group_hash('premise',r['premise']) in dev_groups]
  # Exact duplicate annotations collapse to one row; contradictory gold labels for the
  # same sentence pair are excluded rather than teaching an impossible target.
  rows=dedupe_snli(rows)
  rows=select(rows,min(limit,len(rows)),lambda r:r['premise']+'\0'+r['hypothesis']+'\0'+str(r['label']),lambda r:int(r['label']))
  for r in rows:
   rid=rank('snli',r['premise'],r['hypothesis'],r['label'])[:20];group=group_hash('premise',r['premise']);partition=fit_partition(group) if split=='train' else ('dev' if split=='validation' else 'test')
   state={'premise':r['premise'],'hypothesis':r['hypothesis']};vals=[('entailment','the hypothesis follows'),('neutral','the premise does not determine the hypothesis'),('contradiction','the hypothesis conflicts with the premise')]
   out.append(record('snli',rid,group,split,partition,state,phrase('snli',rid),{'type':'choice','options':options(vals,rid)},mapping[int(r['label'])],lic))
 return out

def candidate_spans(context,question,gold,key,want=4):
 gold=' '.join(gold.split());terms=set(re.findall(r"[\w'-]+",question.lower()));n=max(1,min(6,len(gold.split()) if gold else 2));tokens=re.findall(r"\S+",context)
 pool=[]
 for i in range(max(0,len(tokens)-n+1)):
  text=' '.join(tokens[i:i+n]).strip(' ,.;:!?()[]{}"')
  if text and text.lower()!=gold.lower() and not set(re.findall(r"[\w'-]+",text.lower()))<=terms and len(text)<=128:pool.append(text)
 pool=list(dict.fromkeys(pool));picked=sorted(pool,key=lambda x:rank('span',key,x))[:want-1]
 return ([gold] if gold else [])+picked
def context_window(context,answer_start=None,max_chars=24000):
 if len(context.encode())<=max_chars:return context
 center=answer_start if answer_start is not None and answer_start>=0 else 0;start=max(0,center-max_chars//2);end=min(len(context),start+max_chars);start=max(0,end-max_chars)
 return context[start:end]
def squad_rows(raw=RAW/'squad2/squad_v2',limit_train=17500,limit_test=2000):
 lic=verified_license(LIC/'squad2/DATA-LICENSE-CC-BY-SA-4.0.txt','CC-BY-SA-4.0');out=[]
 for split,path,limit in [('train',raw/'train-00000-of-00001.parquet',limit_train),('validation',raw/'validation-00000-of-00001.parquet',limit_test)]:
  rows=read_rows(path);rows=select(rows,min(limit,len(rows)),lambda r:r['id'],lambda r:not len(r['answers']['text']))
  for r in rows:
   texts=[str(x) for x in r['answers']['text'] if str(x).strip()];starts=[int(x) for x in r['answers']['answer_start']];gold=texts[0] if texts else '';rid=str(r['id']);group=group_hash('context',r['title']+'\0'+r['context']);partition=fit_partition(group) if split=='train' else 'test'
   context=context_window(r['context'],starts[0] if starts else None);spans=candidate_spans(context,r['question'],gold,rid)
   if gold and gold not in spans:raise ValueError(f'SQuAD {rid}: gold span absent')
   if len(spans)<2:continue
   # Candidate IDs are assigned from a gold-independent hash order. Otherwise the
   # construction order would make every answerable row `span_1`, leaking the target.
   ordered_spans=sorted(spans,key=lambda s:rank('span-values',rid,s));vals=[(f'span_{i+1}',s) for i,s in enumerate(ordered_spans)];target=next((v for v,d in vals if d==gold),None)
   q=phrase('squad2',rid)+'\nQuestion: '+r['question'];out.append(record('squad2',rid,group,split,partition,{'title':r['title'],'passage':context},q,{'type':'choice','options':options(vals,rid)},target,lic,'insufficient_evidence' if target is None else None))
 return out

def convert():return fever_rows()+boolq_rows()+snli_rows()+squad_rows()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--report',type=Path,default=REPORT);a=p.parse_args();rows=convert();write_jsonl(a.output,rows)
 counts=collections.Counter((r['source'],r['partition']) for r in rows);targets=collections.Counter((r['source'],'unknown' if r['target'] is None else str(r['target'])) for r in rows)
 report={'records':len(rows),'output':str(a.output),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'counts':{f'{s}/{p}':n for (s,p),n in sorted(counts.items())},
  'targets':{f'{s}/{t}':n for (s,t),n in sorted(targets.items())},'sources':sorted({r['source'] for r in rows}),'phrasing_variants':{s:len(PHRASINGS[s]) for s in PHRASINGS},
  'policies':{'grouping':'passage/context/premise groups never cross partitions; official held-out groups take precedence','calibration':'derived only from official training groups','fever':'cited Wikipedia sentences resolved from official wiki-pages.zip; NOT ENOUGH INFO maps to unknown','squad2':'gold answer plus deterministic same-context n-gram distractors; unanswerable maps to unknown'}}
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
