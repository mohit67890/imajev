"""Post-hoc abstention prior: add an offset to the unknown logit, fit on the calibration fold, evaluate on test and panels.

v2 abstains more than v1.1 (its targets carry the base model's unknown mass and ~30% of teacher-labelled rows are unknown).
This measures how much of that is a recoverable prior shift rather than lost ability.
"""
import json, collections, math, sys
sys.path.insert(0, 'src')
from vision_decision.calibration import TemperatureCalibrator
L = lambda p: [json.loads(l) for l in open(p)]
def predict(r, delta, cal=None):
    raw = dict(r['raw_logits']); raw['unknown'] = raw['unknown'] + delta
    if cal:
        t = cal.temperature(r['decision_type'], r['option_count']) or 1.0
        raw = {k: v / t for k, v in raw.items()}
    m = max(raw.values()); z = sum(math.exp(v - m) for v in raw.values())
    probs = {k: math.exp(v - m) / z for k, v in raw.items()}
    pred = max(probs, key=probs.get)
    keys = list(raw)  # same order as labels; unknown last
    correct = keys.index(pred) == r['target_index']
    if pred == r['prediction']: correct = r['correct']  # trust the recorded verdict when the argmax is unchanged
    return pred, probs[pred], correct
def evaluate(rows, delta, cal=None):
    res = [predict(r, delta, cal) for r in rows]
    acc = sum(c for _, _, c in res) / len(res)
    ece = 0
    for b in range(10):
        sel = [(p, c) for _, p, c in res if b / 10 <= p < (b + 1) / 10 or (b == 9 and p == 1)]
        if sel: ece += len(sel) / len(res) * abs(sum(c for _, c in sel) / len(sel) - sum(p for p, _ in sel) / len(sel))
    unk = sum(p == 'unknown' for p, _, _ in res) / len(res)
    return acc, ece, unk
cal_rows = L('reports/decision-v2/eval/calibration-2b-v2/predictions.jsonl')
test = L('reports/decision-v2/eval/tuned-2b-v2/predictions.jsonl')
v11ids = {json.loads(l)['id'] for l in open('reports/decision-v1.1/eval/tuned-2b/predictions.jsonl')}
test11 = [r for r in test if r['id'] in v11ids]
mmlu = [r for r in test11 if r['source'] == 'mmlu']
exam = L('reports/decision-v2/panels/v1exam-tuned/predictions.jsonl'); held = [r for r in exam if r['group'] == 'held_out_source']
typed = L('reports/decision-v2/panels/typed-tuned/predictions.jsonl'); irr = L('reports/decision-v2/panels/irrelevance-tuned/predictions.jsonl'); sst = L('reports/decision-v2/panels/sst5-tuned/predictions.jsonl')
reas = [r for r in L('reports/decision-v2/panels/reasoning-tuned/predictions.jsonl') if r['source'] != 'typed_decisions_devsel']
print("delta  | calib-fold acc | v1.1-test acc / unk | mmlu | exam held-out | typed | irrelevance | sst5 | authored-reasoning")
best = None
for delta in [0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]:
    ca = evaluate(cal_rows, delta)[0]
    ta, te, tu = evaluate(test11, delta)
    print(f"{delta:5.1f}  | {ca:.1%} | {ta:.1%} / {tu:.1%} | {evaluate(mmlu, delta)[0]:.1%} | {evaluate(held, delta)[0]:.1%} | {evaluate(typed, delta)[0]:.1%} | {evaluate(irr, delta)[0]:.1%} | {evaluate(sst, delta)[0]:.1%} | {evaluate(reas, delta)[0]:.1%}")
    if best is None or ca > best[0]: best = (ca, delta)
print(f"best on calibration fold: delta={best[1]} ({best[0]:.1%})")
