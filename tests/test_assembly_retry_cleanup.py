import json
import pytest
from PIL import Image
from imajev_bench.assemble import assemble

def spec():
    return {'split_salt':'fixture','sources':[
        {'id':'a','path':'a.png','source_cluster':'a','synthetic':True,'provenance':{'generator':'fixture'}},
        {'id':'b','path':'b.png','source_cluster':'b','synthetic':True,'provenance':{'generator':'fixture'}}], 'items':[
        {'id':name,'track':'visual','family':'color','images':[name],'field':{'id':'q','type':'boolean','question':'Red?'},'draft_gold':True} for name in ('a','b')]}

def test_missing_second_asset_cleans_partial_output_and_allows_retry(tmp_path):
    Image.new('RGB',(4,4),'red').save(tmp_path/'a.png')
    output=tmp_path/'out'
    with pytest.raises(FileNotFoundError): assemble(spec(),tmp_path,output)
    assert not output.exists()
    Image.new('RGB',(4,4),'blue').save(tmp_path/'b.png')
    receipt=assemble(spec(),tmp_path,output)
    assert receipt['records']==2 and (output/'records.jsonl').is_file()
    assert len(list((output/'assets').iterdir()))==2

def test_existing_output_is_never_removed(tmp_path):
    output=tmp_path/'out'; output.mkdir(); (output/'important.txt').write_text('preserve')
    with pytest.raises(FileExistsError): assemble(spec(),tmp_path,output)
    assert (output/'important.txt').read_text()=='preserve'

def test_invalid_item_after_first_copy_is_cleaned(tmp_path):
    Image.new('RGB',(4,4),'red').save(tmp_path/'a.png')
    data=spec(); data['items'][1]['typo']=True
    with pytest.raises(ValueError,match='unknown keys'): assemble(data,tmp_path,tmp_path/'out')
    assert not (tmp_path/'out').exists()
