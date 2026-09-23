"""Verify and compare four direct-scoring development runs."""
import json, hashlib, math, statistics
from pathlib import Path
from vision_decision.contracts import Request, Result
from vision_decision.scoring import compile_question, combine_rotations
ROOT=Path(__file__).resolve().parents[2]
folders={'Qwen 2B':'imajev-bench-qwen2b-base-v1','imajev 1.1 2B':'imajev-bench-imajev-v1.1','Gemma 4 E4B':'imajev-bench-gemma-v1','Qwen 9B':'imajev-bench-qwen9b-base-v1'}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def eq(a,b):return type(a) is type(b) and a==b
runs={};summaries={}
for model,folder in folders.items():
 p=ROOT/'reports'/folder;cfg=read(p/'config.json');done=read(p/'completion.json')
 assert sha(p/'predictions.jsonl')==done['predictions_sha256']
 assert sha(p/'runner-used.py')==cfg['runner_sha256']
 assert sha(p/'backend-used.py')==cfg['backend_sha256']
 rows=[json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()]
 assert len(rows)==done['n']==113
 assert len({(r['suite'],r['id']) for r in rows})==113
 suites={}
 for suite,h in cfg['datasets'].items():
  source=ROOT/'data/imajev-bench'/suite/'records.jsonl';assert sha(source)==h
  refs=[json.loads(x) for x in source.read_text().splitlines()]
  rs=[r for r in rows if r['suite']==suite];assert [r['id'] for r in rs]==[r['id'] for r in refs]
  for r,ref in zip(rs,refs):
   assert eq(r['gold'],ref['gold']) and r['correct']==eq(r['prediction'],ref['gold'])
   assert r['image_hashes']==[i['sha256'] for i in ref['images']]
   assert all(sha(source.parent/i['path'])==i['sha256'] for i in ref['images'])
   req=Request.model_validate(ref['request']);_,choices,_=compile_question(req.fields[0],req.state)
   Result.model_validate(r['result'])
   assert eq(combine_rotations(choices,r['passes']).value,r['prediction'])
   assert r['prediction'] is None or any(eq(r['prediction'],c[0]) for c in choices)
   if model.startswith('Qwen'):
    assert cfg['adapter'] is None
    assert all(m['readout']=='language_model_rows' and len(m['image_grid_thw'])==r['image_count'] for m in r['metadata'])
  ts=sorted(sum(m['preprocess_seconds']+m['forward_seconds'] for m in r['metadata']) for r in rs)
  suites[suite]=dict(n=len(rs),correct=sum(r['correct'] for r in rs),p50=statistics.median(ts),p95=ts[math.ceil(.95*len(ts))-1],correct_unknown=sum(r['gold'] is None and r['prediction'] is None for r in rs),false_abstention=sum(r['gold'] is not None and r['prediction'] is None for r in rs))
 runs[model]=dict(rows=rows,config=cfg)
 summaries[model]=dict(suites=suites,valid_outputs=len(rows),peak_gib=done['peak_memory_bytes']/1024**3,load_seconds=done['load_seconds'])
base=runs['Qwen 2B']
for model,run in runs.items():
 assert run['config']['datasets']==base['config']['datasets'] and run['config']['rotations']==4
 for a,b in zip(run['rows'],base['rows']):
  assert a['id']==b['id'] and a['image_hashes']==b['image_hashes']
  assert [m['prompt'] for m in a['metadata']]==[m['prompt'] for m in b['metadata']]
  if model=='Qwen 9B':
   for ma,mb in zip(a['metadata'],b['metadata']):
    assert ma['image_grid_thw']==mb['image_grid_thw']
manifest=read(ROOT/'data/imajev-bench/interventions-v1/manifest.json')
for model,run in runs.items():
 byid={r['id']:r for r in run['rows']};groups=[]
 for g in manifest['groups']:
  a,b,c=[byid[g[k]] for k in ('base','flip','invariant')]
  groups.append(dict(id=g['group_id'],all_correct=all(x['correct'] for x in (a,b,c)),relevant_changed=not eq(a['prediction'],b['prediction']),irrelevant_unchanged=eq(a['prediction'],c['prediction'])))
 summaries[model]['triplets']={k:sum(g[k] for g in groups) for k in ('all_correct','relevant_changed','irrelevant_unchanged')}
 summaries[model]['groups']=groups
pairs={}
for model in ('Qwen 2B','imajev 1.1 2B','Gemma 4 E4B'):
 pairs[model]={'qwen9b_only_correct':sum(a['correct'] and not b['correct'] for a,b in zip(runs['Qwen 9B']['rows'],runs[model]['rows'])),'other_only_correct':sum(b['correct'] and not a['correct'] for a,b in zip(runs['Qwen 9B']['rows'],runs[model]['rows']))}
p=ROOT/'reports/imajev-bench-qwen9b-base-v1'
(p/'comparison.json').write_text(json.dumps(dict(models=summaries,paired_vs_9b=pairs),indent=2)+'\n')
lines=['# Four-model comparison on imajev-bench development cases','','Identical 113 frozen cases, images, semantic prompts and up to four cyclic option rotations. Native model templates/processors; no fitted calibration. Qwen 9B is untuned with no imajev adapter.','','| Suite | '+' | '.join(folders)+' |','| --- | '+' | '.join(['---:']*4)+' |']
for suite in base['config']['datasets']:
 cells=[]
 for d in summaries.values():
  s=d['suites'][suite];cells.append(f'{s["correct"]}/{s["n"]} ({s["correct"]/s["n"]:.1%})')
 lines.append('| '+suite+' | '+' | '.join(cells)+' |')
lines+=['','## Image-change sensitivity (12 groups)','','| Metric | '+' | '.join(folders)+' |','| --- | '+' | '.join(['---:']*4)+' |']
for k in ('all_correct','relevant_changed','irrelevant_unchanged'):
 lines.append('| '+k+' | '+' | '.join(f'{d["triplets"][k]}/12' for d in summaries.values())+' |')
lines+=['','## Latency: p50 / p95 seconds','','Sum of preprocessing and inference across rotations, excluding model loading, image disk reads, network and serialization. Full uncached forwards, not optimized shared-prefix serving. One run per case, nearest-rank p95.','','| Suite | '+' | '.join(folders)+' |','| --- | '+' | '.join(['---:']*4)+' |']
for suite in base['config']['datasets']:
 lines.append('| '+suite+' | '+' | '.join(f'{d["suites"][suite]["p50"]:.2f} / {d["suites"][suite]["p95"]:.2f}' for d in summaries.values())+' |')
lines+=['','## Output validity, memory and abstention','','| Model | Valid outputs | Peak MLX memory | Correct Unknown (18) | False abstention (95) |','| --- | ---: | ---: | ---: | ---: |']
for model,d in summaries.items():
 lines.append(f'| {model} | {d["valid_outputs"]}/113 | {d["peak_gib"]:.2f} GiB | {sum(s["correct_unknown"] for s in d["suites"].values())}/18 | {sum(s["false_abstention"] for s in d["suites"].values())}/95 |')
lines+=['','Type safety is enforced by direct option selection and typed serialization for all models. It is not a free-form JSON generation test. Unknown means an explicit insufficient-evidence option; scores are uncalibrated.','','## Validation and limits','','Verified dataset and output hashes, unique case coverage, option-domain and result-schema validity, prediction recomputation from saved logits, matching cross-model prompts and image hashes. Qwen 2B and 9B processor image grids match. All model snapshots and runner/backend source hashes are pinned in run configuration.','','Small provisional development sets with shared photos and diagram templates; no independent-trial statistical or held-out leaderboard claim. Model size, architecture and training differ; this comparison cannot establish size as the sole cause. Gemma uses different image preprocessing.','']
for model,v in pairs.items(): lines.append(f'Against {model}: Qwen 9B alone correct on {v["qwen9b_only_correct"]} cases; {model} alone correct on {v["other_only_correct"]}.')
(p/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
