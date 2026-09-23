"""Check ordered image delivery with distinguishable benchmark speed signs."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import json
from pathlib import Path
from PIL import Image
from vision_decision.gemma_backend import GemmaDirect
from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template
root=Path('data/imajev-bench/real-pilot-v1/assets')
a=Image.open(root/'abc123e417f1.jpg').convert('RGB')
b=Image.open(root/'023facf36328.jpg').convert('RGB')
model=GemmaDirect('artifacts/model-gemma.json')
rows=[]
for images,position,expected in [([a,b],'first','10'),([a,b],'second','50'),([b,a],'first','50'),([b,a],'second','10')]:
    prompt=f'Look at the {position} image. What speed limit numeral does its sign show? Answer only 10 or 50.'
    rendered=apply_chat_template(model.processor,model.model.config,prompt,num_images=2,enable_thinking=False)
    answer=generate(model.model,model.processor,rendered,image=images,max_tokens=8,temperature=0,verbose=False).text.strip()
    row=dict(order='10,50' if images[0] is a else '50,10',position=position,expected=expected,answer=answer,passed=answer==expected)
    rows.append(row);print(row,flush=True)
Path('reports/imajev-bench-gemma-v1/multi-image-controls.json').write_text(json.dumps(rows,indent=2)+'\n')
if not all(r['passed'] for r in rows): raise RuntimeError('Image-order controls failed')
