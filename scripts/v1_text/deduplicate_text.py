"""Remove exact state overlap from lower-priority folds before augmentation."""
import collections, hashlib, json
PRIORITY={'train':0,'calibration':1,'dev':2,'test':3}
def state_key(row):
 state=row['request'].get('state')
 if isinstance(state,dict) and set(state)=={'text'}:state=state['text']
 value=' '.join(state.split()).casefold() if isinstance(state,str) else json.dumps(state,sort_keys=True,ensure_ascii=False)
 return hashlib.sha256(value.encode()).hexdigest() if value not in ('','{}','[]','null') else None

def deduplicate(rows):
 top={}
 for r in rows:
  key=state_key(r)
  if key:top[key]=max(top.get(key,-1),PRIORITY[r['partition']])
 kept=[];removed=collections.Counter()
 for r in rows:
  key=state_key(r)
  if key and PRIORITY[r['partition']]<top[key]:removed[f"{r['source']}/{r['partition']}"]+=1
  else:kept.append(r)
 return kept,dict(removed)
