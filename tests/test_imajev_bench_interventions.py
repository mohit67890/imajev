import copy,importlib.util,json,re,sys
from pathlib import Path
from PIL import Image,ImageChops
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/imajev_bench'))
from build_interventions import build
from analyze_interventions import check_groups,summarize
from imajev_bench.runner import digest
from imajev_bench.schema import model_payload


def test_triplets_reproduce_and_pixel_changes_are_local(tmp_path):
    rows,m=build(tmp_path/'a');other,_=build(tmp_path/'b');assert rows==other
    check_groups(rows,m);byid={r['id']:r for r in rows}
    for g in m['groups']:
        images=[Image.open(tmp_path/'a'/byid[g[k]]['images'][0]['path']) for k in ('base','flip','invariant')]
        flip=ImageChops.difference(images[0],images[1]).getbbox();same=ImageChops.difference(images[0],images[2]).getbbox()
        assert flip and same
        if g['family']=='numeric_sum':
            assert 400<=flip[0]<flip[2]<=635 and 230<=flip[1]<flip[3]<=305
            assert 400<=same[0]<same[2]<=635 and 325<=same[1]<same[3]<=400
        else:
            assert same==(645,420,681,451)
            assert flip[2]-flip[0]<=65 and flip[3]-flip[1]<=65
            for k,im in zip(('base','flip','invariant'),images):
                r=byid[g[k]];n=sum(im.getpixel((97+(i%6)*110,207+(i//6)*115))==(36,102,204) for i in range(12));rule=r['request']['state']['rule']
                if g['family']=='count_threshold':
                    threshold=int(re.search(r'at (?:most|least) (\d+)',rule)[1]);answer=n<=threshold if 'at most' in rule else n>=threshold
                else:
                    lo=int(re.search(r'at most (\d+)',rule)[1]);hi=int(re.search(r'through (\d+)',rule)[1]);answer=1 if n<=lo else 2 if n<=hi else 3
                assert type(answer) is type(r['gold']) and answer==r['gold']


def test_relation_scoring_requires_correct_endpoints(tmp_path):
    rows,m=build(tmp_path/'a');ex={'protocol':'imajev-modality-probe-v1','predictions':[{'id':r['id'],'input_sha256':digest(model_payload(r)),'value':None,'evidence':'control'} for r in rows]}
    result=summarize(rows,m,ex)
    assert result['overall']['irrelevant_unchanged']==12
    assert result['overall']['irrelevant_pair_correct']==0
    assert result['overall']['all_three_correct']==0
    changed=copy.deepcopy(rows);changed[1]['request']['state']={'different':'rule'}
    with pytest.raises(ValueError,match='Nonimage'):check_groups(changed,m)


def test_packet_does_not_include_labels_or_group_metadata(tmp_path):
    rows,m=build(tmp_path/'a');p=json.loads((tmp_path/'a'/'blind.json').read_text())
    assert len(p['items'])==36
    assert all(set(r)=={'id','input_sha256','request','images'} for r in p['items'])
