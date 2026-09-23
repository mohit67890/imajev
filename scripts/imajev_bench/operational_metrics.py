"""Descriptive operational metrics from completed local direct-scoring runs."""
import json,math,statistics
from pathlib import Path
from vision_decision.contracts import Result
root=Path(__file__).resolve().parents[2]
models={'imajev 1.1 2B':'imajev-bench-imajev-v1.1','Qwen3.5-2B base':'imajev-bench-qwen2b-base-v1','Gemma 4 E4B':'imajev-bench-gemma-v1'}
metrics={}
for name,folder in models.items():
 p=root/'reports'/folder
 rows=[json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()]
 complete=json.loads((p/'completion.json').read_text())
 valid=0
 for r in rows:
  Result.model_validate(r['result'])
  refs=[json.loads(x) for x in (root/'data/imajev-bench'/r['suite']/'records.jsonl').read_text().splitlines()]
  field=next(x for x in refs if x['id']==r['id'])['request']['fields'][0]
  v=r['prediction']
  ok=v is None or (type(v) is bool if field['type']=='boolean' else type(v) is str and v in [o['value'] for o in field['options']] if field['type']=='choice' else type(v) is int and v in [o['value'] for o in field['levels']])
  assert ok;valid+=1
 d={'completed':len(rows),'type_and_domain_valid':valid,'peak_gib':complete['peak_memory_bytes']/1024**3,'load_seconds':complete['load_seconds'],'suites':{}}
 for suite in sorted({r['suite'] for r in rows}):
  rs=[r for r in rows if r['suite']==suite]
  times=sorted(sum(m['preprocess_seconds']+m['forward_seconds'] for m in r['metadata']) for r in rs)
  d['suites'][suite]={'n':len(rs),'p50_seconds':statistics.median(times),'p95_seconds_nearest_rank':times[math.ceil(.95*len(times))-1],'correct_unknown':sum(r['gold'] is None and r['prediction'] is None for r in rs),'gold_unknown':sum(r['gold'] is None for r in rs),'false_abstention':sum(r['gold'] is not None and r['prediction'] is None for r in rs),'answerable':sum(r['gold'] is not None for r in rs)}
 metrics[name]=d
out=root/'reports/imajev-bench-qwen2b-base-v1'
(out/'operational-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
lines=['# Operational metrics for the 113-case development runs','','All three models use direct candidate scoring and typed serialization. Output validity is enforced by this interface; this is not a free-form JSON generation or HTTP API reliability benchmark. Latency sums preprocessing and forward time across up to four uncached rotations, excluding load, disk reads, serialization and network. P95 uses nearest rank; one run per case, no repeated timing trial.','','| Suite | imajev p50 / p95 | Base Qwen p50 / p95 | Gemma p50 / p95 |','| --- | --- | --- | --- |']
for suite in metrics['imajev 1.1 2B']['suites']:
 vals=[f'{d["suites"][suite]["p50_seconds"]:.2f}s / {d["suites"][suite]["p95_seconds_nearest_rank"]:.2f}s' for d in metrics.values()]
 lines.append('| '+suite+' | '+' | '.join(vals)+' |')
lines+=['','| Model | Valid typed outputs | Peak MLX memory | Load time | Correct Unknown | False abstention |','| --- | ---: | ---: | ---: | ---: | ---: |']
for name,d in metrics.items():
 ss=list(d['suites'].values())
 lines.append(f'| {name} | {d["type_and_domain_valid"]}/{d["completed"]} | {d["peak_gib"]:.2f} GiB | {d["load_seconds"]:.2f}s | {sum(s["correct_unknown"] for s in ss)}/{sum(s["gold_unknown"] for s in ss)} | {sum(s["false_abstention"] for s in ss)}/{sum(s["answerable"] for s in ss)} |')
lines+=['','These aggregate abstention counts describe the selected cases; cases are correlated. No cost, throughput, concurrency, optimized serving latency, or calibration claim follows from these runs. Raw predictions and per-pass timing remain in each model run directory.']
(out/'operational-metrics.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
