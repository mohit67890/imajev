"""v2 (2B, pseudo-labelled + self-distilled) against base 2B, 2B v1.1 and 9B v1.1 on every shared panel.

Reads the pod prediction files under reports/decision-v2 and the historical ones; writes reports/decision-v2/eval/v2-comparison.md.
"""
import json, collections, sys, math
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
def brier(rows, conf=lambda r: r['confidence']):
    return sum((conf(r) - r['correct']) ** 2 for r in rows) / len(rows)
cal = TemperatureCalibrator.load('reports/decision-v2/calibration-v2.json')
def calconf(r):
    raw = {('__unknown__' if k == 'unknown' else k): v for k, v in r['raw_logits'].items()}
    s, _ = cal.calibrate_scores(raw, r["decision_type"], r["option_count"])
    return max(s.values())
out = []
P = out.append

# ---- 1. v1.1 test set, identical ids
base = L('reports/decision-v1.1/eval/base-2b/predictions.jsonl'); v11 = L('reports/decision-v1.1/eval/tuned-2b/predictions.jsonl')
v9 = L('reports/decision-v1.1-9b/eval/tuned-9b/predictions.jsonl'); v2 = L('reports/decision-v2/eval/tuned-2b-v2/predictions.jsonl')
b2 = L('reports/decision-v2/eval/base-2b-v2/predictions.jsonl')
ids = sorted(set(v11) & set(v2) & set(v9) & set(base))
P(f"## v1.1 test set, identical {len(ids):,} decisions\n")
P("| | Base 2B | 2B v1.1 | 9B v1.1 | **2B v2 raw** | 2B v2 calibrated |\n|---|---:|---:|---:|---:|---:|")
def row(name, sel):
    P(f"| {name} | {acc([base[i] for i in sel]):.1%} | {acc([v11[i] for i in sel]):.1%} | {acc([v9[i] for i in sel]):.1%} | **{acc([v2[i] for i in sel]):.1%}** | – |")
row('Accuracy (abstention credited)', ids)
img = [i for i in ids if v11[i]['group'] == 'trained_source']; txt = [i for i in ids if v11[i]['group'] != 'trained_source']
row(f'Image decisions ({len(img):,})', img); row(f'Text decisions ({len(txt):,})', txt)
mm = [i for i in ids if v11[i]['source'] == 'mmlu']; row('MMLU in test (never trained)', mm)
ans = [i for i in ids if v11[i]['target'] not in ('unknown', '__unknown__', None)]
fa = lambda M: sum(M[i]['prediction'] in ('unknown', '__unknown__') for i in ans) / len(ans)
P(f"| False abstention on answerable | {fa(base):.1%} | {fa(v11):.1%} | {fa(v9):.1%} | **{fa(v2):.1%}** | – |")
rows2 = [v2[i] for i in ids]
P(f"| ECE (10 bins) | {ece([base[i] for i in ids]):.3f} | {ece([v11[i] for i in ids]):.3f} | {ece([v9[i] for i in ids]):.3f} | **{ece(rows2):.3f}** | {ece(rows2, calconf):.3f} |")
P(f"| Brier | {brier([base[i] for i in ids]):.3f} | {brier([v11[i] for i in ids]):.3f} | {brier([v9[i] for i in ids]):.3f} | **{brier(rows2):.3f}** | {brier(rows2, calconf):.3f} |")
for thr in (0.95,):
    def aa(M, conf=lambda r: r['confidence']):
        sel = [M[i] for i in ids if conf(M[i]) >= thr]; return f"{len(sel)/len(ids):.1%} @ {acc(sel):.1%}" if sel else '–'
    P(f"| Auto-accept at ≥{thr} | – | {aa(v11)} | {aa(v9)} | **{aa(v2)}** | {aa(v2, calconf)} |")
fixed = sum(v2[i]['correct'] and not v11[i]['correct'] for i in ids); broken = sum(v11[i]['correct'] and not v2[i]['correct'] for i in ids)
P(f"\nPaired v1.1 → v2: {fixed:,} fixed, {broken:,} broken.\n")
P("Per source (v1.1 → v2, Δ):\n")
by = collections.defaultdict(list)
for i in ids: by[v11[i]['source']].append(i)
P("| Source | n | 2B v1.1 | 2B v2 | Δ |\n|---|---:|---:|---:|---:|")
for s, sel in sorted(by.items(), key=lambda kv: acc([v2[i] for i in kv[1]]) - acc([v11[i] for i in kv[1]])):
    a, b = acc([v11[i] for i in sel]), acc([v2[i] for i in sel]); P(f"| {s} | {len(sel)} | {a:.1%} | {b:.1%} | {b-a:+.1%} |")

# ---- 2. new-source held-out rows (teacher labels)
new = [i for i in v2 if i not in v11]
P(f"\n## New-source test rows, teacher-labelled ({len(new):,} decisions; accuracy is agreement with the 9B teacher, not with humans)\n")
P("| Source | n | Base 2B | **2B v2** | teacher-unknown rows | v2 abstains on them |\n|---|---:|---:|---:|---:|---:|")
bys = collections.defaultdict(list)
for i in new: bys[v2[i]['source']].append(i)
for s, sel in sorted(bys.items()):
    unk = [i for i in sel if v2[i]['target'] in ('unknown', '__unknown__', None)]
    P(f"| {s} | {len(sel)} | {acc([b2[i] for i in sel]):.1%} | **{acc([v2[i] for i in sel]):.1%}** | {len(unk)} | {sum(v2[i]['prediction'] in ('unknown','__unknown__') for i in unk)} |")
P(f"| all | {len(new)} | {acc([b2[i] for i in new]):.1%} | **{acc([v2[i] for i in new]):.1%}** | | |")

# ---- 3. v1 image exam
P("\n## v1 image exam (identical 24,221 cases)\n")
xb = L('reports/decision-v1/eval/base-2b/predictions.jsonl'); x1 = L('reports/decision-v1/eval/tuned-2b-full/predictions.jsonl')
x11 = L('reports/decision-v1.1/panels/v1exam-tuned/predictions.jsonl'); x9 = L('reports/decision-v1.1-9b/panels/v1exam-tuned/predictions.jsonl'); x2 = L('reports/decision-v2/panels/v1exam-tuned/predictions.jsonl')
xids = sorted(set(xb) & set(x1) & set(x11) & set(x9) & set(x2))
P("| Panel | n | Base 2B | v1 | 2B v1.1 | 9B v1.1 | **2B v2** |\n|---|---:|---:|---:|---:|---:|---:|")
def xrow(name, sel): P(f"| {name} | {len(sel):,} | {acc([xb[i] for i in sel]):.1%} | {acc([x1[i] for i in sel]):.1%} | {acc([x11[i] for i in sel]):.1%} | {acc([x9[i] for i in sel]):.1%} | **{acc([x2[i] for i in sel]):.1%}** |")
xrow('held-out sources', [i for i in xids if xb[i]['group'] == 'held_out_source']); xrow('trained sources', [i for i in xids if xb[i]['group'] != 'held_out_source'])
src = collections.defaultdict(list)
for i in xids:
    if xb[i]['group'] == 'held_out_source': src[xb[i].get('source') or xb[i]['family']].append(i)
for s, sel in sorted(src.items()): xrow(f'· {s}', sel)
unk = [i for i in xids if xb[i]['group'] == 'held_out_source' and xb[i]['target'] == 'unknown']
P(f"\nHeld-out abstention targets n={len(unk)}: v1.1 {sum(x11[i]['correct'] for i in unk)}, 9B {sum(x9[i]['correct'] for i in unk)}, v2 {sum(x2[i]['correct'] for i in unk)} correct.\n")

# ---- 4. text panels
P("## Text panels (never trained on)\n")
P("| Panel | n | 2B v1.1 | 9B v1.1 | **2B v2** (raw ECE / calibrated ECE) |\n|---|---:|---:|---:|---:|")
for name in ('typed', 'sst5', 'irrelevance'):
    t2 = list(L(f'reports/decision-v1.1/panels/{name}-tuned/predictions.jsonl').values()); t9 = list(L(f'reports/decision-v1.1-9b/panels/{name}-tuned/predictions.jsonl').values()); tv = list(L(f'reports/decision-v2/panels/{name}-tuned/predictions.jsonl').values())
    P(f"| {name} | {len(tv)} | {acc(t2):.1%} (ECE {ece(t2):.3f}) | {acc(t9):.1%} (ECE {ece(t9):.3f}) | **{acc(tv):.1%}** ({ece(tv):.3f} / {ece(tv, calconf):.3f}) |")
    if name == 'irrelevance':
        fam2 = collections.defaultdict(list); famv = collections.defaultdict(list)
        for r in t2: fam2[r.get('family')].append(r)
        for r in tv: famv[r.get('family')].append(r)
        mm = [f for f in famv if 'mmlu' in str(f).lower()]
        for f in sorted(famv):
            if 'mmlu' in str(f).lower() or 'text_only' in str(f).lower(): P(f"| · {f} | {len(famv[f])} | {acc(fam2[f]):.1%} | – | **{acc(famv[f]):.1%}** |")

# ---- 5. reasoning dev panel (pod, full manifest)
P("\n## Reasoning dev set on the pod (full manifest, v2 only; v1.1/9B baselines on the authored 240 are in reports/v2-datasets/reasoning-dev-baselines.md)\n")
rv = list(L('reports/decision-v2/panels/reasoning-tuned/predictions.jsonl').values())
fam = collections.defaultdict(list)
for r in rv: fam[r['family']].append(r)
P("| Family | n | 2B v2 |\n|---|---:|---:|")
for f, sel in sorted(fam.items(), key=lambda kv: -len(kv[1])): P(f"| {f} | {len(sel)} | {acc(sel):.1%} |")
typed_sel = [r for r in rv if r['source'] == 'typed_decisions_devsel']; auth = [r for r in rv if r['source'] != 'typed_decisions_devsel']
P(f"| typed-decisions dev-selection | {len(typed_sel)} | {acc(typed_sel):.1%} |\n| authored (240) | {len(auth)} | {acc(auth):.1%} |")
Path('reports/decision-v2/eval/v2-comparison.md').write_text('\n'.join(out) + '\n')
print('\n'.join(out))
