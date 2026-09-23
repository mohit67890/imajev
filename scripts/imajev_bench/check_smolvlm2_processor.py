import os,json,hashlib
os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
from PIL import Image
from mlx_vlm.models.smolvlm.processing_smolvlm import SmolVLMProcessor as AutoProcessor
p=AutoProcessor.from_pretrained(json.loads(Path('artifacts/model-smolvlm2.json').read_text())['path'])
a=Image.open('data/imajev-bench/real-pilot-v1/assets/abc123e417f1.jpg').convert('RGB')
b=Image.open('data/imajev-bench/real-pilot-v1/assets/023facf36328.jpg').convert('RGB')
def prep(images):
 text=p.apply_chat_template([{'role':'user','content':[{'type':'image'} for _ in images]+[{'type':'text','text':'Describe the images.'}]}],add_generation_prompt=True,tokenize=False)
 return p(text=text,images=images,return_tensors='np')
import numpy as np
x,y,xy,yx=[prep(v) for v in ([a],[b],[a,b],[b,a])]
ax=np.array(x['pixel_values'][0]);by=np.array(y['pixel_values'][0])
assert np.array_equal(np.array(xy['pixel_values'][0]),np.concatenate([ax,by]))
assert np.array_equal(np.array(yx['pixel_values'][0]),np.concatenate([by,ax]))
report=dict(single_image_patch_counts=[len(ax),len(by)],pair_patch_count=len(xy['pixel_values'][0]),ordered_pixel_blocks_verified=True,control_content_correct=3,control_content_total=4,limitation='Generation got the first sign wrong for one ordering; other three answers agree after stripping terminal periods. This is a behavioral failure, not proof of missing image inputs. Original raw controls retained.')
Path('reports/imajev-bench-smolvlm2-controls-v1/processor-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
