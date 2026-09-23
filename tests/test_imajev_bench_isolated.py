import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/imajev_bench'))
import run_isolated_codex as runner
from imajev_bench.runner import digest,file_digest


def setup(tmp_path):
    root=tmp_path/'assets';root.mkdir();(root/'image.png').write_bytes(b'pixels')
    payload={'request':{'request_id':'case1','state':{'rule':'blue'},'fields':[{'id':'q','type':'boolean','question':'Blue?'}]},'images':[{'path':'image.png','sha256':file_digest(root/'image.png')}]}
    item={'id':'case1','input_sha256':digest(payload),**payload};out=tmp_path/'out';out.mkdir();return item,root,out


def fake_run(command,**kwargs):
    work=Path(command[command.index('--cd')+1]);assert list(p.name for p in work.iterdir())==['image-0.png']
    assert '--ephemeral' in command and '--ignore-user-config' in command and command[command.index('--sandbox')+1]=='read-only'
    assert 'gold' not in kwargs['input'] and 'group_id' not in kwargs['input']
    assert command[command.index('--image')+1]==str(work/'image-0.png')
    Path(command[command.index('--output-last-message')+1]).write_text(json.dumps({'value':True,'evidence':'Blue pixels'}))
    kwargs['stdout'].write(json.dumps({'type':'thread.started','thread_id':'unique'})+'\n'+json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'answer'}})+'\n')
    return SimpleNamespace(returncode=0)


def test_one_direct_image_request_and_receipts(tmp_path,monkeypatch):
    item,root,out=setup(tmp_path);monkeypatch.setattr(runner.subprocess,'run',fake_run)
    receipt=runner.run_one(item,root,out,'gpt-5.6-sol')
    assert receipt['status']=='complete' and receipt['thread_id']=='unique'
    assert json.loads((out/'case1'/'prediction.json').read_text())['input_sha256']==item['input_sha256']


def test_tool_execution_invalidates_isolation(tmp_path,monkeypatch):
    item,root,out=setup(tmp_path)
    def tool_run(command,**kwargs):
        result=fake_run(command,**kwargs);kwargs['stdout'].write(json.dumps({'type':'item.completed','item':{'type':'command_execution'}})+'\n');return result
    monkeypatch.setattr(runner.subprocess,'run',tool_run)
    receipt=runner.run_one(item,root,out,'gpt-5.6-sol')
    assert receipt['status']=='failed' and not (out/'case1'/'prediction.json').exists()


def test_unknown_event_rejected_and_unsafe_ids_blocked(tmp_path,monkeypatch):
    import pytest
    item,root,out=setup(tmp_path)
    def tool_run(command,**kwargs):
        result=fake_run(command,**kwargs);kwargs['stdout'].write(json.dumps({'type':'item.completed','item':{'type':'future_external_tool'}})+'\n');return result
    monkeypatch.setattr(runner.subprocess,'run',tool_run)
    assert runner.run_one(item,root,out,'gpt-5.6-sol')['status']=='failed'
    item['id']='../escape'
    with pytest.raises(ValueError,match='Unsafe'):runner.run_one(item,root,out,'gpt-5.6-sol')


def test_two_images_keep_attachment_order(tmp_path,monkeypatch):
    item,root,out=setup(tmp_path);(root/'second.png').write_bytes(b'second pixels')
    item['images'].append({'path':'second.png','sha256':file_digest(root/'second.png')});item['input_sha256']=digest({'request':item['request'],'images':item['images']})
    def two_run(command,**kwargs):
        image_args=[Path(command[i+1]) for i,v in enumerate(command) if v=='--image']
        assert [p.read_bytes() for p in image_args]==[b'pixels',b'second pixels']
        Path(command[command.index('--output-last-message')+1]).write_text(json.dumps({'value':True,'evidence':'Both images inspected in order'}))
        kwargs['stdout'].write(json.dumps({'type':'thread.started','thread_id':'two'})+'\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess,'run',two_run)
    receipt=runner.run_one(item,root,out,'gpt-5.6-sol')
    assert receipt['status']=='complete' and receipt['image_sha256_in_order']==[i['sha256'] for i in item['images']]
