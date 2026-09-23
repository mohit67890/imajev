"""v2.1 (2B, state-grounded + pairs data, continued from v2) against 2B v1.1, 2B v2 and 9B v1.1 on every shared panel.

Writes reports/decision-v2.1/eval/v21-comparison.md.
"""
import json, collections, sys
from pathlib import Path
sys.path.insert(0, 'src')
from vision_decision.calibration import TemperatureCalibrator
L = lambda p: {r['id']: r for r in map(json.loads, open(p))}
acc = lambda rows: sum(r['correct'] for r in rows) / len(rows) if rows else float('nan')
def ece(rows, conf=lambda r: r['confidence'], bins=10):
    tot = 0
    for b in range(bins):
        sel = [r for r in rows if b / bins <= conf(r) < (b + 1) / bins or (b == bins - 1 and conf(r) == 1)]
        if sel: tot += len(sel) / len(rows) * abs(acc(sel) - sum(conf(r) for r in sel) / len(sel))
    return tot
calp = Path('reports/decision-v2.1/calibration-v2.1.json'); cal = TemperatureCalibrator.load(calp) if calp.exists() else None
def calconf(r):
    raw = {('__unknown__' if k == 'unknown' else k): v for k, v in r['raw_logits'].items()}
    s, _ = cal.calibrate_scores(raw, r['decision_type'], r['option_count']); return max(s.values())
out = []; P = out.append
M = {'2B v1.1': 'reports/decision-v1.1/eval/tuned-2b/predictions.jsonl', '9B v1.1': 'reports/decision-v1.1-9b/eval/tuned-9b/predictions.jsonl',
     '2B v2': 'reports/decision-v2/eval/tuned-2b-v2/predictions.jsonl', '2B v2.1': 'reports/decision-v2.1/eval/tuned-2b-v21/predictions.jsonl'}
T = {k: L(v) for k, v in M.items()}
ids = sorted(set.intersection(*(set(t) for t in T.values())))
P(f"## v1.1 test set, identical {len(ids):,} decisions\n")
P("| | " + " | ".join(T) + " |\n|---|" + "---:|" * len(T))
def row(name, sel, f=lambda t, sel: f"{acc([t[i] for i in sel]):.1%}"): P(f"| {name} | " + " | ".join(f(t, sel) for t in T.values()) + " |")
row('Accuracy (abstention credited)', ids)
row('Image decisions', [i for i in ids if T['2B v1.1'][i]['group'] == 'trained_source']); row('Text decisions', [i for i in ids if T['2B v1.1'][i]['group'] != 'trained_source'])
row('MMLU in test', [i for i in ids if T['2B v1.1'][i]['source'] == 'mmlu'])
mm = [i for i in ids if T['2B v1.1'][i]['source'] == 'mmlu']
row('MMLU predicted-unknown rate', mm, lambda t, sel: f"{sum(t[i]['prediction'] in ('unknown','__unknown__') for i in sel)/len(sel):.1%}")
ans = [i for i in ids if T['2B v1.1'][i]['target'] not in ('unknown', '__unknown__', None)]
row('False abstention on answerable', ans, lambda t, sel: f"{sum(t[i]['prediction'] in ('unknown','__unknown__') for i in sel)/len(sel):.1%}")
row('ECE raw (10 bins)', ids, lambda t, sel: f"{ece([t[i] for i in sel]):.3f}")
if cal: P(f"| ECE v2.1 calibrated | | | | {ece([T['2B v2.1'][i] for i in ids], calconf):.3f} |")
v2, v21 = T['2B v2'], T['2B v2.1']
P(f"\nPaired v2 → v2.1: {sum(v21[i]['correct'] and not v2[i]['correct'] for i in ids):,} fixed, {sum(v2[i]['correct'] and not v21[i]['correct'] for i in ids):,} broken.\n")
# new-source rows shared by v2 and v2.1 (teacher labels)
new = [i for i in v21 if i in v2 and i not in T['2B v1.1']]
P(f"## v2 new-source test rows shared with v2.1 ({len(new):,}): v2 {acc([v2[i] for i in new]):.1%} → v2.1 {acc([v21[i] for i in new]):.1%}\n")
v21new = [i for i in v21 if i not in v2]
bys = collections.defaultdict(list)
for i in v21new: bys[v21[i]['source']].append(i)
P(f"## v2.1-only test rows ({len(v21new):,}; state_grounded/pairs_natural = teacher agreement, pairs_grounded = constructed labels)\n")
P("| Source | n | 2B v2.1 | abstains |\n|---|---:|---:|---:|")
for s, sel in sorted(bys.items()): P(f"| {s} | {len(sel)} | {acc([v21[i] for i in sel]):.1%} | {sum(v21[i]['prediction'] in ('unknown','__unknown__') for i in sel)/len(sel):.1%} |")
# v1 exam
P("\n## v1 image exam (identical cases)\n")
X = {'2B v1.1': L('reports/decision-v1.1/panels/v1exam-tuned/predictions.jsonl'), '9B v1.1': L('reports/decision-v1.1-9b/panels/v1exam-tuned/predictions.jsonl'),
     '2B v2': L('reports/decision-v2/panels/v1exam-tuned/predictions.jsonl'), '2B v2.1': L('reports/decision-v2.1/panels/v1exam-tuned/predictions.jsonl')}
xb = L('reports/decision-v1/eval/base-2b/predictions.jsonl'); xids = sorted(set(xb) & set.intersection(*(set(x) for x in X.values())))
P("| Panel | n | " + " | ".join(X) + " |\n|---|---:|" + "---:|" * len(X))
def xrow(name, sel): P(f"| {name} | {len(sel):,} | " + " | ".join(f"{acc([x[i] for i in sel]):.1%}" for x in X.values()) + " |")
xrow('held-out sources', [i for i in xids if xb[i]['group'] == 'held_out_source']); xrow('trained sources', [i for i in xids if xb[i]['group'] != 'held_out_source'])
src = collections.defaultdict(list)
for i in xids:
    if xb[i]['group'] == 'held_out_source': src[xb[i].get('source') or xb[i]['family']].append(i)
for s, sel in sorted(src.items()): xrow(f'· {s}', sel)
# text panels + reasoning
P("\n## Text panels and reasoning dev\n")
P("| Panel | n | 2B v1.1 | 9B v1.1 | 2B v2 | 2B v2.1 |\n|---|---:|---:|---:|---:|---:|")
for name in ('typed', 'sst5', 'irrelevance'):
    t = {k: list(L(f'reports/{d}/panels/{name}-tuned/predictions.jsonl').values()) for k, d in (('2B v1.1', 'decision-v1.1'), ('9B v1.1', 'decision-v1.1-9b'), ('2B v2', 'decision-v2'), ('2B v2.1', 'decision-v2.1'))}
    P(f"| {name} | {len(t['2B v2.1'])} | " + " | ".join(f"{acc(v):.1%}" for v in t.values()) + " |")
    if name == 'irrelevance':
        fam = {k: [r for r in v if r.get('family') == 'mmlu_heldout'] for k, v in t.items()}
        P(f"| · mmlu with irrelevant image | {len(fam['2B v2.1'])} | " + " | ".join(f"{acc(v):.1%}" for v in fam.values()) + " |")
for k, d in (('2B v2', 'decision-v2'), ('2B v2.1', 'decision-v2.1')):
    rv = list(L(f'reports/{d}/panels/reasoning-tuned/predictions.jsonl').values())
    auth = [r for r in rv if r['source'] != 'typed_decisions_devsel']; ts = [r for r in rv if r['source'] == 'typed_decisions_devsel']
    P(f"| reasoning dev on pod ({k}): authored 240 / typed-devsel 6000 | | | | {acc(auth):.1%} / {acc(ts):.1%} |" if k == '2B v2' else f"| reasoning dev on pod ({k}) | | | | | {acc(auth):.1%} / {acc(ts):.1%} |")
# probes: pod (step-50) and local
P("\n## Probes\n")
P("| Probe | 2B v1.1 | 9B v1.1 | 2B v2 | 2B v2.1 step 50 (pod) | 2B v2.1 step 50 (Mac) | 2B v2.1 final step 1361 (Mac) |\n|---|---:|---:|---:|---:|---:|---:|")
def probe(m, n):
    p = Path(f'reports/v2-datasets/{n}-probe-{m}.json')
    if not p.exists(): return None
    d = json.load(open(p)); d = d.get('report', d); return d
for n, pod in (('state', 'state-probe-tuned'), ('pairs', 'pairs-probe-tuned')):
    podrows = list(L(f'reports/decision-v2.1/panels/{pod}/predictions.jsonl').values())
    cells = []
    for m in ('2b-v1.1', '9b-v1.1', '2b-v2'):
        d = probe(m, n); cells.append(f"{d['overall_accuracy']:.1%}" if d else '–')
    cells.append(f"{acc(podrows):.1%}")
    for m in ('2b-v2.1-best', '2b-v2.1-last'):
        d = probe(m, n); cells.append(f"{d['overall_accuracy']:.1%}" if d else 'pending')
    P(f"| {n} | " + " | ".join(cells) + " |")
    fam = collections.defaultdict(list)
    for r in podrows: fam[r['family']].append(r)
    for f, sel in sorted(fam.items()):
        cells = []
        for m in ('2b-v1.1', '9b-v1.1', '2b-v2'):
            d = probe(m, n); cells.append(f"{d['families'][f]['accuracy']:.0%}" if d and f in d['families'] else '–')
        cells.append(f"{acc(sel):.0%}")
        for m in ('2b-v2.1-best', '2b-v2.1-last'):
            d = probe(m, n); cells.append(f"{d['families'][f]['accuracy']:.0%}" if d and f in d['families'] else 'pending')
        P(f"| · {f} | " + " | ".join(cells) + " |")
Path('reports/decision-v2.1/eval').mkdir(parents=True, exist_ok=True)
Path('reports/decision-v2.1/eval/v21-comparison.md').write_text('\n'.join(out) + '\n'); print('\n'.join(out))
