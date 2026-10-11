import pytest
from imajev_bench.assemble import assemble

def spec(fractions):
    return {'split_salt':'fixture','split_fractions':fractions,'sources':[], 'items':[
        {'id':'one','track':'text','family':'rule','field':{'id':'q','type':'boolean','question':'Yes?'},'draft_gold':True}]}

@pytest.mark.parametrize('fractions',[{'dev':-1,'calibration':0,'test':2},{'dev':float('nan'),'test':1},{'dev':float('inf'),'test':float('-inf')},{'dev':True,'test':0}, {'dev':'0.2','test':0.8},{}])
def test_invalid_split_fractions_are_rejected_before_output_creation(tmp_path,fractions):
    output=tmp_path/'out'
    with pytest.raises(ValueError,match='split_fractions'):
        assemble(spec(fractions),tmp_path,output)
    assert not output.exists()

@pytest.mark.parametrize('fractions',[{'dev':1,'test':0},{'dev':0.2,'calibration':0.1,'test':0.7},{'test':1}])
def test_valid_zero_and_partial_split_weights_remain_supported(tmp_path,fractions):
    receipt=assemble(spec(fractions),tmp_path,tmp_path/'out')
    assert receipt['records']==1
    assert receipt['split_fractions']==fractions
    assert set(receipt['split_records'])<=set(k for k,v in fractions.items() if v)
