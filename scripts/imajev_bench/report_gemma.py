"""Verify frozen run receipts and summarize Gemma development evidence."""
import json,hashlib
from pathlib import Path
from collections import Counter
p=Path('reports/imajev-bench-gemma-v1')
read=lambda x:json.loads(x.read_text())
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
rows=[json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()]
cfg=read(p/'config.json');done=read(p/'completion.json')
assert sha(p/'predictions.jsonl')==done['predictions_sha256']
assert sha(p/'runner-used.py')==cfg['runner_sha256'] and sha(p/'backend-used.py')==cfg['backend_sha256']
for name,h in cfg['datasets'].items():
    source=Path('data/imajev-bench')/name/'records.jsonl'
    assert sha(source)==h
    records=[json.loads(x) for x in source.read_text().splitlines()]
    rr=[r for r in rows if r['suite']==name]
    assert Counter(r['id'] for r in rr)==Counter(r['id'] for r in records)
    for r,ref in zip(rr,records):
        assert r['id']==ref['id'] and r['image_hashes']==[i['sha256'] for i in ref['images']]
        assert r['correct']==(type(r['prediction']) is type(ref['gold']) and r['prediction']==ref['gold'])
        for m in r['metadata']:
            assert m['image_count']==r['image_count']
            assert (m['visual_tokens']>0)==bool(r['image_count'])
controls=read(p/'multi-image-controls.json')
assert len(controls)==4 and all(c['passed'] for c in controls)
manifest=read(Path('data/imajev-bench/interventions-v1/manifest.json'))
byid={r['id']:r for r in rows if r['suite']=='interventions-v1'}
groups=[]
for g in manifest['groups']:
    a,b,c=[byid[g[k]] for k in ('base','flip','invariant')]
    groups.append(dict(group_id=g['group_id'],relevant_changed=a['prediction']!=b['prediction'],irrelevant_unchanged=a['prediction']==c['prediction'],all_three_correct=all(r['correct'] for r in (a,b,c))))
(p/'paired-results.json').write_text(json.dumps(groups,indent=2)+'\n')
lines=['# Gemma 4 E4B-it on imajev-bench development suites','', 'Pinned local BF16 checkpoint. Native Gemma image processing; up to four cyclic answer orders, averaged candidate log-probabilities. No conversational history or gold passed to the model.','', '| Suite | Reference agreement | Unknown outputs | Median request time |','| --- | ---: | ---: | ---: |']
for name,s in read(p/'summary.json').items():lines.append(f'| {name} | {s["correct"]}/{s["n"]} ({s["correct"]/s["n"]:.1%}) | {s["unknown_predictions"]} | {s["median_seconds"]:.2f}s |')
lines+=['',f'Peak MLX memory: {done["peak_memory_bytes"]/1024**3:.2f} GiB. Model load: {done["load_seconds"]:.1f}s. Timings are one local run including repeated image processing across rotations; not a controlled speed comparison.','',f'Controlled triplets: {sum(g["all_three_correct"] for g in groups)}/12 entirely correct; {sum(g["relevant_changed"] for g in groups)}/12 predictions changed after the relevant edit; {sum(g["irrelevant_unchanged"] for g in groups)}/12 remained unchanged after the irrelevant edit. Invariance alone does not imply correct reasoning.','', '## Validation and limits','', 'All 113 IDs, frozen dataset hashes, output hash, image hashes, and processor image presence checked. 37 schema/scoring/harder-real tests passed. All four image-order generation controls passed, including reversed image order; saved in multi-image-controls.json. These are smoke checks, not full cross-framework parity tests.','', 'All suites are development drafts with provisional labels. Photos recur across cases and suites; controlled diagrams share templates. Do not interpret the pooled total as independent observations or a held-out leaderboard. Earlier fashion Qwen results are on a different dataset and cannot be directly compared here. Candidate probabilities are uncalibrated. This run measures direct option scoring, not unrestricted generation.','', 'The initial sandbox attempt failed before inference because Metal was unavailable; its configuration is preserved in ../imajev-bench-gemma-v1-sandbox-init-failed. A subsequent partial run stopped after 96 cases on a multi-image metadata shape error; preserved in ../imajev-bench-gemma-v1-partial-metadata-error. The final run restarted every case after the logging fix and used local GPU access in offline mode.']
(p/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
