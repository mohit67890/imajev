"""Score the authored reasoning-dev items with an MLX bundle (+optional adapter): accuracy per family, abstention credited."""
import sys,json,time,collections
sys.path[:0]=['src','scripts']
from vision_decision.backend import MLXDirect
from decision_data import expand_fields
from vision_decision.contracts import Request
bundle,adapter,label=sys.argv[1],(None if sys.argv[2]=='none' else sys.argv[2]),sys.argv[3]
rows=[json.loads(l) for l in open('data/manifests/decision-v2-reasoning-dev.jsonl')]
rows=[r for r in rows if r['source']=='reasoning_authored']
engine=MLXDirect(bundle,adapter=adapter);fam=collections.defaultdict(lambda:[0,0]);t=time.time();n=0
for r in rows:
    for item in expand_fields(r):
        req=Request.model_validate(item['request'])
        res=engine.score_request(None,req.fields,req.state,rotations=1)
        out=res[0][0]
        pred=out.value
        want=item['target']
        # normalise: ordinal levels compare as str; booleans as bool
        ok=(pred is None and want is None) or (pred is not None and want is not None and str(pred)==str(want))
        fam[r['family']][0]+=1;fam[r['family']][1]+=ok;n+=1
tot=sum(v[1] for v in fam.values());print(f"{label}: authored {tot}/{n} = {tot/n:.1%}  ({time.time()-t:.0f}s)  "+" ".join(f"{k}={v[1]}/{v[0]}" for k,v in sorted(fam.items())))
