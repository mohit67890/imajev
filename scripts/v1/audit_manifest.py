"""Quality audit of the assembled decision-v1 manifest: shortcuts a model could exploit, duplicates, broken files, odd records."""
import json,random,sys
from pathlib import Path
from collections import Counter,defaultdict
from PIL import Image
rows=[json.loads(l) for l in Path('data/manifests/decision-v1.jsonl').read_text().splitlines()];F=lambda r:r['request']['fields'][0]
by=defaultdict(list)
for r in rows:by[r['source']].append(r)
report={};flags=[]
def values(f):return [o['value'] for o in f.get('options',[])] or [l['value'] for l in f.get('levels',[])]
missing=[im['image'] for r in rows for im in r['images'] if not Path(im['image']).is_file()]
if missing:flags.append(f'{len(missing)} image files missing, e.g. {missing[:2]}')
rng=random.Random(0);bad=0
for r in rng.sample(rows,3000):
 for im in r['images']:
  try:
   with Image.open(im['image']) as x:x.load();assert x.mode in('RGB','L') and min(x.size)>=16
  except Exception:bad+=1
if bad:flags.append(f'{bad}/3000 sampled images unreadable or tiny')
for s,rs in sorted(by.items()):
 fit=[r for r in rs if r['partition']!='test'];test=[r for r in rs if r['partition']=='test'];info={}
 # 1. duplicate (image, question) pairs
 keys=Counter((tuple(im['sha256'] for im in r['images']),F(r)['question']) for r in rs);info['duplicate_questions']=sum(n-1 for n in keys.values() if n>1)
 # 2. how far a blind guesser gets on the test split using only training answer frequencies (no image)
 prior=Counter(str(r['target']) for r in fit);typed=defaultdict(list)
 for r in test or fit[:2000]:
  f=F(r);cands=[str(v) for v in ([True,False] if f['type']=='boolean' else values(f))]+['None']
  guess=max(cands,key=lambda v:prior.get(v,0));typed[f['type']].append(guess==str(r['target']))
 info['blind_prior_accuracy']={t:round(sum(v)/len(v),3) for t,v in typed.items()}
 chance=defaultdict(list)
 for r in test or fit[:2000]:
  f=F(r);chance[f['type']].append(1/(3 if f['type']=='boolean' else len(values(f))+1))
 info['uniform_chance']={t:round(sum(v)/len(v),3) for t,v in chance.items()}
 # 3. does the question text alone reveal abstention? (same question string always/never unknown)
 q=defaultdict(list)
 for r in fit:q[F(r)['question']].append(r['target'] is None)
 rep=[v for v in q.values() if len(v)>=5];info['questions_asked_5plus_times']=len(rep);info['of_which_always_unknown']=sum(all(v) for v in rep)
 # 4. gold position in stored order (test is served in stored order)
 pos=Counter()
 for r in test:
  f=F(r)
  if f['type']=='choice' and r['target'] is not None:pos[round(values(f).index(r['target'])/max(1,len(values(f))-1),1)]+=1
 if pos:info['gold_first_or_last_share_in_test']=round((pos[0.0]+pos[1.0])/sum(pos.values()),3)
 # 5. awkward values
 info['options_named_unknown_yes_no']=sum(1 for r in rs if F(r)['type']=='choice' and any(str(v).strip().lower() in('unknown','yes','no','none','n/a') for v in values(F(r))))
 info['empty_or_short_questions']=sum(len(F(r)['question'].strip())<8 for r in rs)
 info['unknown_share']=round(sum(r['target'] is None for r in fit)/max(1,len(fit)),3);info['train']=len(fit)
 report[s]=info
 for t,a in info['blind_prior_accuracy'].items():
  if a>max(.6,info['uniform_chance'][t]+.3) and not s.startswith('heldout_'):flags.append(f'{s}/{t}: a blind guesser scores {a:.0%} (chance {info["uniform_chance"][t]:.0%}) — answers predictable without the image')
 if info['duplicate_questions']>len(rs)*.01:flags.append(f'{s}: {info["duplicate_questions"]} duplicate image+question pairs')
 if info['options_named_unknown_yes_no']>len(rs)*.01:flags.append(f'{s}: {info["options_named_unknown_yes_no"]} choice records with an option literally named unknown/yes/no')
 if info['of_which_always_unknown']>max(3,.05*max(1,info['questions_asked_5plus_times'])):flags.append(f'{s}: {info["of_which_always_unknown"]} repeated question strings are ALWAYS unknown — wording leaks abstention')
Path('reports/decision-v1/audit.json').write_text(json.dumps(dict(flags=flags,sources=report),indent=1)+'\n')
print(f'{len(rows)} records, {len(by)} sources, all image files present: {not missing}\n');print('FLAGS:' if flags else 'NO FLAGS');print('\n'.join(' - '+f for f in flags))
print('\nsource | train | unknown | blind-prior accuracy (chance)')
for s,i in report.items():print(f" {s:22s} {i['train']:7d}  {i['unknown_share']:.2f}   "+'  '.join(f"{t}:{a:.2f}({i['uniform_chance'][t]:.2f})" for t,a in i['blind_prior_accuracy'].items()))
