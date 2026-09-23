"""Audit every expanded text decision with the training tokenizer/chat template."""
import argparse, collections, json, math, hashlib
from PIL import Image
from pathlib import Path
from functools import lru_cache
from transformers import AutoProcessor
from decision_data import expand_fields, render, pixel_budget
from vision_decision.scoring import readout_codes

def main():
 p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--report',type=Path,required=True);p.add_argument('--max-length',type=int,default=4096);a=p.parse_args()
 bundle=json.loads(Path('artifacts/model.json').read_text());processor=AutoProcessor.from_pretrained(bundle['path'],local_files_only=True,trust_remote_code=False)
 @lru_cache(None)
 def wrapper(n_images):
  marker='__IMAJEV_PROMPT_SLOT_973__'
  rendered=processor.apply_chat_template([{'role':'user','content':[{'type':'image'}]*n_images+[{'type':'text','text':marker}]}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
  before,found,after=rendered.partition(marker)
  if not found or marker in after:raise ValueError('Unexpected prompt template')
  return before,after
 def template(prompt,n_images=0):
  before,after=wrapper(n_images);return before+prompt+after

 codes=[c for c,_ in readout_codes(processor.tokenizer,template(''),255)]
 rejected=[];count=0;max_tokens=0;images=0;hist=collections.Counter();image_sizes={};visual_counts={}
 for line in a.manifest.open():
  row=json.loads(line)
  if row.get('images'):images+=1
  sizes=[]
  for im in row.get('images',[]):
   path=im['image']
   if path not in image_sizes:
    with Image.open(path) as image:image_sizes[path]=image.size
   width,height=image_sizes[path];scale=min(1,math.sqrt(pixel_budget(row,400000)/(width*height)))
   sizes.append((max(1,int(height*scale)),max(1,int(width*scale))))
  size_key=tuple(sizes)
  if size_key not in visual_counts:
   visual_counts[size_key]=sum(processor._get_num_multimodal_tokens(image_sizes=sizes)['num_image_tokens']) if sizes else 0
  for item in expand_fields(row):
   header,choices,texts,_=render(item)
   prompt=header+'\n'.join(f'{c}: {t}' for c,t in zip(codes,texts))
   n=len(processor.tokenizer.encode(template(prompt,len(sizes)),add_special_tokens=False))+visual_counts[size_key]-len(sizes);count+=1;max_tokens=max(max_tokens,n);hist[str((n//512)*512)]+=1
   if n>a.max_length:rejected.append({'id':item['id'],'record_id':row['id'],'tokens':n,'partition':row['partition']})
 report={'manifest_sha256':hashlib.file_digest(a.manifest.open('rb'),'sha256').hexdigest(),'ok':not rejected,'checked_decisions':count,'max_tokens':max_tokens,'limit':a.max_length,'overlength':rejected,'histogram_512_tokens':dict(hist),'image_requests_checked':images,'scope':'Training chat template and tokenizer, actual image file dimensions, training pixel resize, and installed processor image-token accounting. Runtime augmentation/option permutations retain trainer max_length guard.'}
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='overlength'}));raise SystemExit(0 if report['ok'] else 1)
if __name__=='__main__':main()
