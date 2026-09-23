import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/imajev_bench'))
from build_harder_real import build
from imajev_bench.runner import file_digest


def test_reused_evidence_is_one_component_and_model_packet_is_blind(tmp_path):
    src=tmp_path/'source';(src/'assets').mkdir(parents=True);sources=[]
    for sid in ('a','b','c'):
        image=src/'assets'/f'{sid}.jpg';image.write_bytes(sid.encode())
        sources.append({'source':{'source_id':sid,'creator':'author','license':'CC0','license_url':'https://example.com/license','file_page':'https://example.com/'+sid},'local_image':{'path':'assets/'+image.name,'sha256':file_digest(image)}})
    (src/'collection.json').write_text(json.dumps({'sources':sources}));(src/'records.jsonl').write_text('source receipt')
    cases=[{'source_ids':pair,'family':'comparison','state':{'rule':'compare both images'},'field':{'id':'q','type':'boolean','question':'Accept?'},'gold':True,'evidence':'private draft rationale'} for pair in [('a','b'),('b','c')]]
    path=tmp_path/'cases.json';path.write_text(json.dumps(cases));rows,manifest=build(src,[path],tmp_path/'out')
    assert len({r['group_id'] for r in rows})==1 and manifest['two_image_cases']==2
    packet=json.loads((tmp_path/'out'/'blind.json').read_text())
    assert all(set(i)=={'id','input_sha256','images','request'} for i in packet['items'])
    assert 'private draft rationale' not in json.dumps(packet)
    assert [x['path'] for x in rows[1]['images']]==['assets/b.jpg','assets/c.jpg']


def test_multi_image_credits_are_preserved(tmp_path):
    from imajev_bench.review import build_review
    row={'id':'r','group_id':'g','track':'joint','request':{'request_id':'r','state':{},'fields':[{'id':'q','type':'boolean','question':'Accept?'}]},'images':[],'provenance':{'image_sources':[{'creator':'First creator','license':'CC BY 4.0'},{'creator':'Second creator','license':'CC BY-SA 4.0'}]}}
    row.update(family='comparison',split='dev',gold=True,annotation_status='draft')
    for index in (0,1):
        p=tmp_path/f'{index}.png';p.write_bytes(b'asset'+str(index).encode());row['images'].append({'path':p.name,'sha256':file_digest(p)})
    html=build_review([row],tmp_path,tmp_path/'review.html').read_text()
    assert 'First creator' in html and 'Second creator' in html
