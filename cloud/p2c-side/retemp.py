import json, sys, math
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
summ=json.load(open(sys.argv[1].replace("results.jsonl","summary.json")))
def ece(rows,T):
    conf=[];cor=[]
    for r in rows:
        p=r["probs"]; lg={k:math.log(max(v,1e-12))/T for k,v in p.items()}; m=max(lg.values()); z=sum(math.exp(v-m) for v in lg.values())
        q={k:math.exp(v-m)/z for k,v in lg.items()}; k=max(q,key=q.get); conf.append(q[k]); cor.append(k==r["predicted"] and r["correct"])
    n=len(rows); e=0
    for b in range(10):
        idx=[i for i,c in enumerate(conf) if (b/10<c<=(b+1)/10) or (b==0 and c==0)]
        if idx: e+=len(idx)/n*abs(sum(cor[i] for i in idx)/len(idx)-sum(conf[i] for i in idx)/len(idx))
    return e, sum(c>0.9 for c in conf)
print("summary ece", summ["ece"].get("value", summ["ece"]) if isinstance(summ["ece"],dict) else summ["ece"], "acc", summ["accuracy"])
for T in (1.0,1.3,1.6,2.0,2.3,2.6,3.0):
    e,hi=ece(rows,T); print(f"T={T:.1f} ece {e:.3f}  items>0.9 conf {hi}")
