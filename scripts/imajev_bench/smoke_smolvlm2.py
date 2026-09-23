import os,json
os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
from PIL import Image
from vision_decision.smolvlm_backend import SmolVLMDirect
from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template
b=SmolVLMDirect('artifacts/model-smolvlm2.json')
from mlx.utils import tree_flatten
print('dtypes',sorted({str(p.dtype) for _,p in tree_flatten(b.model.parameters())}),flush=True)
images=[Image.open('data/imajev-bench/real-pilot-v1/assets/abc123e417f1.jpg').convert('RGB')]
rows=[]
for ims,prompt,choices in [(images,'What speed limit numeral is visible? Return only A or B.\nA: 10\nB: 50',[('10','10'),('50','50')]),([], 'What is 1 + 1? Return only A or B.\nA: 2\nB: 3',[(2,'2'),(3,'3')])]:
 # Include the standard Unknown candidate for Result contract validity.
 prompt+='\nC: unknown';choices.append(('__unknown__','unknown'))
 result,meta=b.score_compiled(ims,prompt,['A','B','C'],choices)
 rendered=apply_chat_template(b.processor,b.model.config,prompt,num_images=len(ims),enable_thinking=False)
 answer=generate(b.model,b.processor,rendered,image=ims or None,max_tokens=4,temperature=0,verbose=False).text.strip()
 row=dict(result=result.model_dump(),metadata=meta,generated=answer)
 rows.append(row);print(json.dumps(row),flush=True)
 assert answer in ('A', 'The answer is A') and result.value==choices[0][0]
Path('reports/smolvlm2-smoke.json').write_text(json.dumps(rows,indent=2)+'\n')
