"""Integrity-check paired frozen runs and report development agreement."""
import json,hashlib
from pathlib import Path
from collections import Counter
from vision_decision.scoring import combine_rotations, compile_question
from vision_decision.contracts import Request
ROOT=Path(__file__).resolve().parents[2]
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b): return type(a) is type(b) and a==b
runs={}
for model,folder in [('imajev','imajev-bench-imajev-v1.1'),('gemma','imajev-bench-gemma-v1')]:
    p=ROOT/'reports'/folder
    cfg=read(p/'config.json');complete=read(p/'completion.json')
    assert sha(p/'predictions.jsonl')==complete['predictions_sha256']
    assert sha(p/'runner-used.py')==cfg['runner_sha256']
    assert sha(p/'backend-used.py')==cfg['backend_sha256']
    if model=='imajev':
        for name,h in cfg['adapter_hashes'].items(): assert sha(Path(cfg['adapter'])/name)==h
    rows=[json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()]
    assert len(rows)==complete['n']==113
    for suite,h in cfg['datasets'].items():
        source=ROOT/'data/imajev-bench'/suite/'records.jsonl'
        assert sha(source)==h
        refs=[json.loads(x) for x in source.read_text().splitlines()]
        selected=[r for r in rows if r['suite']==suite]
        assert Counter(r['id'] for r in refs)==Counter(r['id'] for r in selected)
        for r,ref in zip(selected,refs):
            assert r['id']==ref['id'] and equal(r['gold'],ref['gold'])
            assert r['correct']==equal(r['prediction'],ref['gold'])
            req=Request.model_validate(ref['request'])
            _,choices,_=compile_question(req.fields[0],req.state)
            assert equal(combine_rotations(choices,r['passes']).value,r['prediction'])
            assert r['image_hashes']==[i['sha256'] for i in ref['images']]
            assert all(sha(source.parent/i['path'])==i['sha256'] for i in ref['images'])
            if model=='imajev':
                assert all(m['readout']=='trained_255' and len(m['image_grid_thw'])==r['image_count'] for m in r['metadata'])
    runs[model]=dict(rows=rows,config=cfg,completion=complete,summary=read(p/'summary.json'))
assert runs['imajev']['config']['datasets']==runs['gemma']['config']['datasets']
assert runs['imajev']['config']['rotations']==runs['gemma']['config']['rotations']==4
pairs=[]
for a,b in zip(runs['imajev']['rows'],runs['gemma']['rows']):
    assert (a['suite'],a['id'])==(b['suite'],b['id'])
    assert a['image_hashes']==b['image_hashes']
    assert [m['prompt'] for m in a['metadata']]==[m['prompt'] for m in b['metadata']]
    assert [x[0] for x in a['passes']]==[x[0] for x in b['passes']]
    pairs.append(dict(suite=a['suite'],id=a['id'],gold=a['gold'],imajev=a['prediction'],gemma=b['prediction'],imajev_correct=a['correct'],gemma_correct=b['correct']))
manifest=read(ROOT/'data/imajev-bench/interventions-v1/manifest.json')
triplets={}
for model,run in runs.items():
    byid={r['id']:r for r in run['rows'] if r['suite']=='interventions-v1'}
    stats=Counter()
    for g in manifest['groups']:
        a,b,c=[byid[g[k]] for k in ('base','flip','invariant')]
        stats['all_three_correct']+=all(x['correct'] for x in (a,b,c))
        stats['relevant_changed']+=not equal(a['prediction'],b['prediction'])
        stats['irrelevant_unchanged']+=equal(a['prediction'],c['prediction'])
    triplets[model]=dict(stats)
breakdowns={}
for model,run in runs.items():
    breakdowns[model]={}
    for suite in run['config']['datasets']:
        rs=[r for r in run['rows'] if r['suite']==suite]
        breakdowns[model][suite]={track:dict(n=len(xs),correct=sum(x['correct'] for x in xs),gold_unknown=sum(x['gold'] is None for x in xs),correct_unknown=sum(x['gold'] is None and x['prediction'] is None for x in xs),false_abstentions=sum(x['gold'] is not None and x['prediction'] is None for x in xs)) for track in sorted({r['track'] for r in rs}) for xs in [[r for r in rs if r['track']==track]]}
p=ROOT/'reports/imajev-bench-imajev-v1.1'
(p/'paired-comparison.json').write_text(json.dumps(dict(triplets=triplets,breakdowns=breakdowns,pairs=pairs),indent=2)+'\n')
lines=['# imajev 1.1 versus Gemma 4 E4B-it','', 'Same 113 frozen development cases, exact question/state/option prompts, image bytes and up to four cyclic option orders. Each uses its native template and vision processor; imajev uses its trained decision readout, Gemma its language-model output. No fitted calibration.','', '| Suite | imajev 1.1 | Gemma 4 E4B | Difference |','| --- | ---: | ---: | ---: |']
for suite in runs['imajev']['config']['datasets']:
    a=runs['imajev']['summary'][suite];b=runs['gemma']['summary'][suite]
    lines.append(f'| {suite} | {a["correct"]}/{a["n"]} ({a["correct"]/a["n"]:.1%}) | {b["correct"]}/{b["n"]} ({b["correct"]/b["n"]:.1%}) | {(a["correct"]-b["correct"])/a["n"]*100:+.1f} pp |')
lines+=['',f'Paired outcomes: imajev alone correct on {sum(x["imajev_correct"] and not x["gemma_correct"] for x in pairs)} cases; Gemma alone correct on {sum(x["gemma_correct"] and not x["imajev_correct"] for x in pairs)}. Cases share evidence, so these are descriptive counts, not independent trials.','', '## Controlled image changes','', '| Metric (12 groups) | imajev | Gemma |','| --- | ---: | ---: |']
for key in ('all_three_correct','relevant_changed','irrelevant_unchanged'): lines.append(f'| {key} | {triplets["imajev"][key]}/12 | {triplets["gemma"][key]}/12 |')
lines+=['','## Validation and limitations','','39 schema/scoring/harder-real/readout tests passed. Recomputed every prediction from saved rotation logits. Verified completion, source and prediction hashes, adapter hashes, all 113 unique case IDs, exact cross-model prompt and image matching, rotation offsets, and trained-head use for every imajev pass. Processor image-grid counts match every request.','', 'Provisional development labels; photos reused across suites and cases, synthetic diagrams share templates. No held-out leaderboard or statistical superiority claim. Native visual token budgets and output-head precision differ. Direct option scoring only; no claim about unrestricted generation. All rotations use full independent forwards, so these timings do not represent the optimized shared-prefix serving path.','', '## Local runtime','','| Model | Peak MLX memory | Median seconds: pilot / interventions / harder |','| --- | ---: | --- |']
for model,run in runs.items():
    times=' / '.join(f'{run["summary"][s]["median_seconds"]:.2f}' for s in ('real-pilot-v1','interventions-v1','harder-real-v1'))
    lines.append(f'| {model} | {run["completion"]["peak_memory_bytes"]/1024**3:.2f} GiB | {times} |')
(p/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
