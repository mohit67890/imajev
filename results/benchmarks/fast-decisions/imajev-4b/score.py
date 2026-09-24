"""Score preds-*.jsonl: per-domain head accuracy (card rule: every head counted), plus the macro average over domains."""
import json, sys, collections, math

path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
dom = collections.defaultdict(lambda: [0, 0])
head = collections.defaultdict(lambda: [0, 0])
secs = []
for r in rows:
    secs.append(r["seconds"])
    for h in r["heads"]:
        dom[r["domain"]][0] += h["correct"]; dom[r["domain"]][1] += 1
        k = f"{r['domain']}.{h['task']}" + (" (multi)" if h["multi_label"] else "")
        head[k][0] += h["correct"]; head[k][1] += 1
print(f"rows {len(rows)}  median s/row {sorted(secs)[len(secs)//2]:.2f}")
accs = []
for d in sorted(dom):
    c, n = dom[d]; accs.append(c / n)
    print(f"{d:20s} {100*c/n:5.1f}%  ({c}/{n} heads)")
H = sum(v[1] for v in dom.values()); C = sum(v[0] for v in dom.values())
print(f"\nMACRO avg over {len(accs)} domains: {100*sum(accs)/len(accs):.1f}%   pooled heads: {100*C/H:.1f}% ({C}/{H})")
p = C / H
print(f"pooled 95% CI ±{196*math.sqrt(p*(1-p)/H):.1f}")
print("\nper head:")
for k in sorted(head):
    c, n = head[k]; print(f"  {k:40s} {100*c/n:5.1f}%  ({c}/{n})")
