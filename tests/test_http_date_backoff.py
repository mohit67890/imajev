from datetime import datetime,timezone
from email.utils import format_datetime
import pytest
import requests
from v2 import common

class Session:
    def __init__(self,header): self.header=header; self.calls=0
    def get(self,*args,**kwargs):
        self.calls+=1
        response=requests.Response(); response.status_code=503 if self.calls==1 else 200
        response._content=b''
        if self.calls==1: response.headers['Retry-After']=self.header
        return response

@pytest.mark.parametrize('header,expected',[
    (format_datetime(datetime.fromtimestamp(1010,timezone.utc),usegmt=True),10.0),
    (format_datetime(datetime.fromtimestamp(1120,timezone.utc),usegmt=True),30.0),
    (format_datetime(datetime.fromtimestamp(990,timezone.utc),usegmt=True),0.0),
    ('5',5.0),('invalid',1.0),('nan',1.0),('-1',1.0)])
def test_retry_after_dates_and_numeric_values_use_bounded_waits(monkeypatch,header,expected):
    sleeps=[]; monkeypatch.setattr(common.time,'time',lambda:1000)
    monkeypatch.setattr(common.time,'sleep',sleeps.append)
    session=Session(header)
    response=common.get_with_backoff(session,'https://example.invalid/fixture',tries=2)
    assert response.status_code==200 and session.calls==2
    assert sleeps==[expected]

def test_success_does_not_sleep(monkeypatch):
    sleeps=[]; monkeypatch.setattr(common.time,'sleep',sleeps.append)
    session=Session('5'); session.calls=1
    assert common.get_with_backoff(session,'https://example.invalid/fixture').status_code==200
    assert sleeps==[]
