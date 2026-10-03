import hashlib
import base64
import re
import json
import pytest
from PIL import Image
from imajev_bench.review import build_review
from imajev_bench.runner import digest
from imajev_bench.schema import model_payload

def record(tmp_path):
    Image.new('RGB', (4, 4), 'red').save(tmp_path/'image.png')
    blob=(tmp_path/'image.png').read_bytes()
    return {'id':'one','group_id':'g','track':'visual','family':'color','split':'dev','gold':True,'annotation_status':'draft','provenance':{},'images':[{'path':'image.png','sha256':hashlib.sha256(blob).hexdigest()}],
            'request':{'request_id':'one','state':{},'fields':[{'id':'q','type':'boolean','question':'Is it red?'}]}},blob

def test_changed_image_is_rejected_before_packet_output(tmp_path):
    row,_=record(tmp_path)
    Image.new('RGB',(4,4),'blue').save(tmp_path/'image.png')
    output=tmp_path/'review.html'
    with pytest.raises(ValueError,match='Image changed'):
        build_review([row],tmp_path,output)
    assert not output.exists()

def test_intact_image_bytes_and_input_hash_remain_bound(tmp_path):
    row,blob=record(tmp_path)
    html=build_review([row],tmp_path,tmp_path/'review.html').read_text()
    match=re.search(r"const records=JSON\.parse\([^']*'([^']+)'",html)
    packet=json.loads(base64.b64decode(match.group(1)))[0]
    assert base64.b64decode(packet['images'][0].split(',')[1])==blob
    assert packet['input_sha256']==digest(model_payload(row))
