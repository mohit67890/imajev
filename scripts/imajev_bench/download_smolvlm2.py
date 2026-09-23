import os,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
os.environ['HF_HOME']=str(root/'.cache/huggingface')
from huggingface_hub import HfApi,snapshot_download
repo='HuggingFaceTB/SmolVLM2-2.2B-Instruct'
info=HfApi().model_info(repo,files_metadata=True)
size=sum(f.size or 0 for f in info.siblings)
assert size<12*1024**3
print(f'{repo}@{info.sha}: {size} bytes',flush=True)
path=snapshot_download(repo,revision=info.sha,max_workers=4)
p=root/'artifacts/model-smolvlm2.json'
with p.open('x') as f:json.dump(dict(repo=repo,revision=info.sha,path=path,repository_bytes=size),f,indent=2)
print(path,flush=True)
