import json, math, sys
from pathlib import Path

import pytest
from PIL import Image

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from decision_data import expand_fields, load_image, pixel_budget, render
from vision_decision.scoring import readout_codes

def select_cases(rows):
 candidates=[]
 for row in rows:
  if not row.get('images'):continue
  try:sizes=[Image.open(x['image']).size for x in row['images']]
  except (FileNotFoundError,OSError):continue
  candidates.append((row,sizes,len(json.dumps(row['request'],ensure_ascii=False))))
 one=[x for x in candidates if len(x[1])==1];two=[x for x in candidates if len(x[1])==2]
 picks=[min(one,key=lambda x:x[1][0][0]*x[1][0][1]),max(one,key=lambda x:x[1][0][0]*x[1][0][1]),
        min(one,key=lambda x:x[2]),max(one,key=lambda x:x[2]),min(two,key=lambda x:sum(w*h for w,h in x[1])),
        max(two,key=lambda x:sum(w*h for w,h in x[1])),min(two,key=lambda x:x[2]),max(two,key=lambda x:x[2])]
 assert len({x[0]['id'] for x in picks})==8
 return [x[0] for x in picks]

def compare(processor,row,codes):
 sizes=[];images=[]
 for im in row['images']:
  with Image.open(im['image']) as image:width,height=image.size
  scale=min(1,math.sqrt(pixel_budget(row,400000)/(width*height)));sizes.append((max(1,int(height*scale)),max(1,int(width*scale))))
  images.append(load_image(im['image'],pixel_budget(row,400000)))
 item=expand_fields(row)[0];header,choices,texts,_=render(item);prompt=header+'\n'.join(f'{c}: {t}' for c,t in zip(codes,texts))
 def template(text):return processor.apply_chat_template([{'role':'user','content':[{'type':'image'}]*len(images)+[{'type':'text','text':text}]}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
 rendered=template(prompt);text_tokens=len(processor.tokenizer.encode(rendered,add_special_tokens=False))
 visual=sum(processor._get_num_multimodal_tokens(image_sizes=sizes)['num_image_tokens'])
 analytical=text_tokens+visual-len(images)
 exact=int(processor(text=[rendered],images=images,return_tensors='pt')['input_ids'].shape[-1])
 return {'id':row['id'],'source':row['source'],'images':len(images),'original_sizes':[list(Image.open(x['image']).size) for x in row['images']],
         'training_sizes':[list(x) for x in sizes],'text_template_tokens':text_tokens,'visual_tokens':visual,'analytical_tokens':analytical,'processor_tokens':exact,'difference':analytical-exact}

def run_check(manifest=ROOT/'data/manifests/decision-v1.1-draft.jsonl'):
 transformers=pytest.importorskip('transformers');bundle=json.loads((ROOT/'artifacts/model.json').read_text());path=Path(bundle['path'])
 if not path.is_dir() or not manifest.is_file():pytest.skip('local model or draft manifest absent')
 processor=transformers.AutoProcessor.from_pretrained(path,local_files_only=True,trust_remote_code=False)
 template=processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':''}]}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
 codes=[c for c,_ in readout_codes(processor.tokenizer,template,255)];rows=[json.loads(x) for x in manifest.open() if x.strip()]
 return [compare(processor,row,codes) for row in select_cases(rows)]

def test_analytical_image_token_count_matches_actual_processor():
 results=run_check();assert len(results)==8 and all(x['difference']==0 for x in results)
