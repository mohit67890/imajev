"""Frozen development suites, imajev 1.1 native direct scoring, four cyclic orders."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import json, hashlib, shutil
from pathlib import Path
from statistics import median
from PIL import Image
from vision_decision.backend import MLXDirect
from vision_decision.contracts import Request
from vision_decision.scoring import compile_question, cyclic_offsets, rotate, combine_rotations
from imajev_bench.schema import validate_records
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT=ROOT/'reports/imajev-bench-imajev-v1.1'
OUT.mkdir(exist_ok=False)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,indent=2)+'\n')
ADAPTER=ROOT/'reports/decision-v1.1/runs/h100x4/best-mlx'
suites={}
for name in ('real-pilot-v1','interventions-v1','harder-real-v1'):
    root=ROOT/'data/imajev-bench'/name
    suites[name]=(root,validate_records([json.loads(x) for x in (root/'records.jsonl').read_text().splitlines()],root))
save(OUT/'config.json',dict(adapter=str(ADAPTER),adapter_hashes={p.name:sha(p) for p in ADAPTER.iterdir() if p.is_file()},model=json.loads(Path('artifacts/model.json').read_text()),datasets={n:sha(r/'records.jsonl') for n,(r,_) in suites.items()},rotations=4,protocol='Native Qwen processor with imajev 1.1 adapter and trained readout, original dataset image pixels, direct option logits averaged over up to four cyclic orders; no calibration. Gold excluded from model prompt. Draft development data.',runner_sha256=sha(Path(__file__)),backend_sha256=sha(Path('src/vision_decision/backend.py'))))
shutil.copy(__file__,OUT/'runner-used.py')
shutil.copy('src/vision_decision/backend.py',OUT/'backend-used.py')
b=MLXDirect('artifacts/model.json', adapter=str(ADAPTER))
assert b.readout is not None, 'Missing trained decision head'
print(f'Loaded in {b.load_seconds:.1f}s',flush=True)
rows=[]
with (OUT/'predictions.jsonl').open('x') as f:
    for name,(root,records) in suites.items():
        for r in records:
            req=Request.model_validate(r['request'])
            images=[]
            for ref in r['images']:
                path=root/ref['path']
                assert sha(path)==ref['sha256']
                with Image.open(path) as im: images.append(im.convert('RGB'))
            header,choices,texts=compile_question(req.fields[0],req.state)
            labels=b._labels(header,len(choices),len(images))
            passes=[]; metadata=[]
            for offset in cyclic_offsets(len(choices),4):
                prompt=header+'\n'.join(f'{label}: {text}' for label,text in zip(labels,rotate(texts,offset)))
                result,meta=b.score_compiled(images,prompt,labels,rotate(choices,offset))
                assert meta['readout']=='trained_255'
                passes.append((offset,list(result.raw_logits.values())))
                metadata.append(dict(offset=offset,prompt=prompt,**meta))
                if meta['peak_memory_bytes']>40*1024**3: raise RuntimeError('40 GiB memory budget exceeded')
            result=combine_rotations(choices,passes)
            pred=result.value
            row=dict(suite=name,id=r['id'],group_id=r['group_id'],track=r['track'],type=req.fields[0].type,image_count=len(images),image_hashes=[i['sha256'] for i in r['images']],gold=r['gold'],prediction=pred,correct=type(pred) is type(r['gold']) and pred==r['gold'],result=result.model_dump(),passes=passes,metadata=metadata)
            rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
            print(f'{len(rows)}/113 {name} {r["id"]}: {pred!r} correct={row["correct"]}',flush=True)
            b.mx.clear_cache()
summary={}
for name in suites:
    rs=[r for r in rows if r['suite']==name]
    summary[name]=dict(n=len(rs),correct=sum(r['correct'] for r in rs),unknown_predictions=sum(r['prediction'] is None for r in rs),gold_unknown=sum(r['gold'] is None for r in rs),median_seconds=median(sum(m['preprocess_seconds']+m['forward_seconds'] for m in r['metadata']) for r in rs))
save(OUT/'summary.json',summary)
save(OUT/'completion.json',dict(n=len(rows),predictions_sha256=sha(OUT/'predictions.jsonl'),load_seconds=b.load_seconds,peak_memory_bytes=max(m['peak_memory_bytes'] for r in rows for m in r['metadata'])))
print(json.dumps(summary,indent=2),flush=True)
