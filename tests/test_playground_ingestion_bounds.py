import asyncio
import json
import base64
import pytest
from fastapi.testclient import TestClient
from playground import server
from test_playground_server import FakeBackend,REQUEST,png_bytes,data_url

class JSONRequest:
    headers={'content-type':'application/json'}
    def __init__(self,payload,extra=''):
        self.chunks=[json.dumps(payload).encode()+extra.encode(),b' ',b' ']; self.reads=0
    async def body(self):
        self.reads=len(self.chunks); return b''.join(self.chunks)
    async def stream(self):
        for chunk in self.chunks:
            self.reads+=1; yield chunk

def test_json_stream_stops_before_collecting_unbounded_body(monkeypatch):
    monkeypatch.setattr(server,'MAX_REQUEST_BYTES',32,raising=False)
    request=JSONRequest({'questions':{}},extra=' '*40)
    with pytest.raises(server.PlaygroundError) as caught: asyncio.run(server.read_payload(request))
    assert caught.value.status==413 and caught.value.error=='request_too_large'
    assert request.reads==1

def test_excess_images_are_rejected_before_any_base64_decode(monkeypatch):
    calls=[]
    monkeypatch.setattr(server,'decode_data_url',lambda *args:calls.append(args) or b'image')
    with pytest.raises(server.PlaygroundError) as caught:
        asyncio.run(server.read_payload(JSONRequest({'questions':{},'images':['a','b','c']})))
    assert caught.value.status==422 and 'at most two' in caught.value.detail
    assert calls==[]

def test_oversized_base64_is_rejected_before_decoding(monkeypatch):
    monkeypatch.setattr(server,'MAX_BYTES',8)
    value='data:image/png;base64,'+base64.b64encode(b'x'*10).decode()
    calls=[]
    monkeypatch.setattr(server.base64,'b64decode',lambda *args,**kwargs:calls.append(args) or b'x'*10)
    with pytest.raises(server.PlaygroundError) as caught: server.decode_data_url(value,0)
    assert caught.value.status==413 and calls==[]

def test_uploaded_image_copy_is_bounded(monkeypatch):
    monkeypatch.setattr(server,'MAX_BYTES',8)
    class Upload:
        calls=[]
        async def read(self,size=-1): self.calls.append(size); return b'x'*(9 if size<0 else min(size,9))
    upload=Upload()
    class Form:
        def get(self,key): return json.dumps({'questions':{}}) if key=='request' else None
        def getlist(self,key): return [upload] if key=='image' else []
    class Request:
        headers={'content-type':'multipart/form-data'}
        async def form(self): return Form()
    with pytest.raises(server.PlaygroundError) as caught: asyncio.run(server.read_payload(Request()))
    assert caught.value.status==413 and upload.calls==[9]

def test_valid_json_and_multipart_requests_still_reach_backend():
    backend=FakeBackend(); client=TestClient(server.create_app(backend,examples=[]))
    response=client.post('/v1/systemone',json={**REQUEST,'images':[data_url(png_bytes())]})
    assert response.status_code==200 and backend.calls[-1][0]==1
    response=client.post('/v1/systemone',data={'request':json.dumps(REQUEST)},files={'image':('a.png',png_bytes(),'image/png')})
    assert response.status_code==200 and backend.calls[-1][0]==1


def test_embedded_multipart_images_count_before_upload_copies(monkeypatch):
    calls=[]
    class Upload:
        async def read(self,size=-1): calls.append(size); return b'abc'
    class Form:
        def get(self,key):
            return json.dumps({'questions':{},'state':'data:image/png;base64,QUJD'}) if key=='request' else None
        def getlist(self,key): return [Upload(),Upload()] if key=='image' else []
    class Request:
        headers={'content-type':'multipart/form-data'}
        async def form(self): return Form()
    with pytest.raises(server.PlaygroundError) as caught: asyncio.run(server.read_payload(Request()))
    assert caught.value.status==422 and calls==[]

def test_uploaded_request_document_read_is_bounded(monkeypatch):
    monkeypatch.setattr(server,'MAX_REQUEST_BYTES',8)
    calls=[]
    class Upload:
        async def read(self,size=-1): calls.append(size); return b'x'*9
    class Form:
        def get(self,key): return Upload() if key=='request' else None
        def getlist(self,key): return []
    class Request:
        headers={'content-type':'multipart/form-data'}
        async def form(self): return Form()
    with pytest.raises(server.PlaygroundError) as caught: asyncio.run(server.read_payload(Request()))
    assert caught.value.status==413 and calls==[9]
