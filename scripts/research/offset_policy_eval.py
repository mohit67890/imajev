"""Compare unknown-offset policies for the 2B on text panels: none, NLL-fitted, and bounded fits."""
import json, sys, math, collections
sys.path.insert(0, 'src')
from vision_decision.calibration import TemperatureCalibrator, calibration_key, fit_unknown_offset_and_temperature, softmax
R = 'reports/imajev-bench-v2-lite-v1/h100-run/out'
L = lambda p: [json.loads(l) for l in open(p)]
fold = L(f'{R}/2b-calfold/predictions.jsonl'); panels = {'mmlu': L(f'{R}/2b-mmlu/predictions.jsonl'), 'typed': L(f'{R}/2b-typed/predictions.jsonl'), 'sst5': L(f'{R}/2b-sst5/predictions.jsonl')}
def apply(rows, offsets, temps):
    out = []
    for r in rows:
        k = calibration_key(r['decision_type'], r['option_count']); lg = list(r['logits']); lg[-1] += offsets.get(k, 0.0)
        p = softmax(lg, temps.get(k, 1.0)); i = max(range(len(p)), key=lambda j: p[j])
        out.append((i == r['target_index'], p[i], i == len(lg) - 1, r['target_index'] == len(lg) - 1))
    return out
def summarise(res):
    n = len(res); acc = sum(c for c, _, _, _ in res) / n; unk = sum(u for _, _, u, _ in res) / n
    tgt = [x for x in res if x[3]]; ans = [x for x in res if not x[3]]
    ece = 0
    for b in range(10):
        sel = [(c, p) for c, p, _, _ in res if b/10 <= p < (b+1)/10 or (b == 9 and p == 1)]
        if sel: ece += len(sel)/n * abs(sum(c for c, _ in sel)/len(sel) - sum(p for _, p in sel)/len(sel))
    return f"acc {acc:.1%} unk-pred {unk:.1%} correct-abstain {sum(c for c,_,_,_ in tgt)}/{len(tgt)} false-abstain {sum(u for _,_,u,_ in ans)/max(len(ans),1):.1%} ECE {ece:.3f}"
grouped = collections.defaultdict(list)
for r in fold: grouped[calibration_key(r['decision_type'], r['option_count'])].append((r['logits'], r['target_index']))
policies = {}
policies['none (T=1)'] = ({}, {})
fitted = TemperatureCalibrator.load('reports/decision-v2.1/calibration-v2.1-final.json'); policies['NLL fit, unbounded'] = (dict(fitted.unknown_offsets), dict(fitted.temperatures))
for bound in (-1.0, -1.5, -2.0):
    offs, temps = {}, {}
    for k, s in grouped.items():
        o, t = fit_unknown_offset_and_temperature(s, offsets=[x/10 for x in range(int(bound*10), 21)]); offs[k], temps[k] = o, t
    policies[f'NLL fit, offset >= {bound}'] = (offs, temps)
offs_t, temps_t = {}, {}
for k, s in grouped.items(): offs_t[k], temps_t[k] = 0.0, fit_unknown_offset_and_temperature(s, offsets=[0.0])[1]
policies['temperature only'] = (offs_t, temps_t)
for name, (offs, temps) in policies.items():
    print(f"== {name}: offsets {{{', '.join(f'{k}:{v:+.1f}' for k,v in sorted(offs.items()))}}}")
    print(f"   fold  {summarise(apply(fold, offs, temps))}")
    for pn, rows in panels.items(): print(f"   {pn:5s} {summarise(apply(rows, offs, temps))}")
print("fold target-unknown share by bucket:", {k: f"{sum(t==len(l)-1 for l,t in s)/len(s):.0%}" for k, s in grouped.items()})
