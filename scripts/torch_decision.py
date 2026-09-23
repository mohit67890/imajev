"""PyTorch twin of the MLX decision path, for cloud GPUs. Same prompt, same decision position, same float32 candidate head."""
from pathlib import Path
import json
import torch
from vision_decision.scoring import readout_codes, verified_label_ids

TARGETS=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','in_proj_qkv','in_proj_z','out_proj']

class TorchDecision:
 def __init__(self,path,device,dtype=torch.bfloat16,max_length=4096,pad_multiple=0):
  from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
  self.processor=AutoProcessor.from_pretrained(path,local_files_only=True);self.device=device;self.max_length=max_length;self.pad_multiple=pad_multiple  # pad batches to a multiple (fewer distinct kernel shapes)
  self.model=Qwen3_5ForConditionalGeneration.from_pretrained(path,local_files_only=True,dtype=dtype).to(device).eval()
  self.readout=None;self._codebook=None
 def add_lora(self,rank=16,alpha=32,targets=None):
  from peft import LoraConfig,get_peft_model
  # Language layers only, as in the MLX run; the vision tower stays frozen. `targets` narrows the module list (e.g. attention only).
  self.model=get_peft_model(self.model,LoraConfig(r=rank,lora_alpha=alpha,lora_dropout=0.0,target_modules=r'.*language_model.*\.('+'|'.join(targets or TARGETS)+')'))
  return self.model
 def labels(self,count,n_images=0):
  if self._codebook is None:self._codebook=readout_codes(self.processor.tokenizer,self.render('',n_images),255)
  return [x[0] for x in self._codebook[:count]]
 def enable_readout(self,adapter=None,trainable=True):
  """Install the v1.1 255-row head, initialized from the matching LM-head rows.

  Older adapters omit decision_readout.safetensors and continue through vocabulary rows.
  """
  from safetensors.torch import load_file
  base=self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
  if self._codebook is None:self._codebook=readout_codes(self.processor.tokenizer,self.render('',0),255)
  ids=[x[1] for x in self._codebook]
  weight=base.lm_head.weight[torch.tensor(ids,device=self.device)].detach().float().clone()
  if adapter is not None:
   path=Path(adapter)/'decision_readout.safetensors'
   if path.exists():weight=load_file(str(path),device=str(self.device))['weight'].float()
   else:return False
   manifest_path=Path(adapter)/'decision_readout.json'
   if not manifest_path.exists():raise ValueError('Trained readout is missing decision_readout.json tokenizer binding')
   manifest=json.loads(manifest_path.read_text())
   actual=[{'code':c,'token_id':i} for c,i in self._codebook]
   if manifest.get('version')!=1 or manifest.get('codes')!=actual:raise ValueError('Decision readout code/token binding does not match this tokenizer')
  if tuple(weight.shape)!=(255,base.lm_head.weight.shape[1]) or not bool(torch.isfinite(weight).all()):raise ValueError('Decision readout must be finite with shape [255, hidden_size]')
  self.readout=torch.nn.Linear(weight.shape[1],255,bias=False,device=self.device,dtype=torch.float32)
  self.readout.weight.data.copy_(weight);self.readout.weight.requires_grad_(trainable)
  return True
 def save_readout(self,directory):
  if self.readout is None:raise ValueError('Decision readout is not enabled')
  from safetensors.torch import save_file
  save_file({'weight':self.readout.weight.detach().cpu().contiguous()},str(Path(directory)/'decision_readout.safetensors'))
  (Path(directory)/'decision_readout.json').write_text(json.dumps({'version':1,'codes':[{'code':c,'token_id':i} for c,i in self._codebook]},indent=2)+'\n')
 def render(self,prompt,n_images):
  messages=[dict(role='user',content=[dict(type='image')]*n_images+[dict(type='text',text=prompt)])]
  rendered=self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,enable_thinking=False)
  if not rendered.endswith('<think>\n\n</think>\n\n'):raise ValueError('Unexpected Qwen non-thinking template boundary')
  return rendered
 def prepare(self,images,prompt,labels):
  images=[] if images is None else images if isinstance(images,list) else [images];rendered=self.render(prompt,len(images))
  token_ids=verified_label_ids(self.processor.tokenizer,rendered,labels)
  inputs=self.processor(text=[rendered],images=images or None,return_tensors='pt')
  if inputs['input_ids'].shape[-1]>self.max_length:raise ValueError(f'Processed request exceeds the {self.max_length}-token limit')
  suffix=self.processor.tokenizer.encode('</think>\n\n',add_special_tokens=False)
  if inputs['input_ids'][0,-len(suffix):].tolist()!=suffix:raise ValueError('Processed decision-position suffix mismatch')
  return rendered,inputs,token_ids
 def candidate_logits(self,inputs,token_ids):
  inputs={k:v.to(self.device) for k,v in inputs.items()};base=self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
  hidden=base.model(**inputs).last_hidden_state[0,-1]
  indices=self._readout_indices(token_ids)
  return self.readout(hidden.float()) [indices] if self.readout is not None and indices is not None else hidden.float()@base.lm_head.weight[torch.tensor(token_ids,device=self.device)].float().T
 def _readout_indices(self,token_ids):
  if self._codebook is None:return None
  lookup={token:i for i,(_,token) in enumerate(self._codebook)}
  return [lookup[x] for x in token_ids] if all(x in lookup for x in token_ids) else None

 # ---- batched path (keeps large GPUs busy): left padding puts every decision position at index -1 ----
 def render_example(self,images,prompt,labels):
  """CPU-only half of prepare(): safe to run in DataLoader workers."""
  images=[] if images is None else images if isinstance(images,list) else [images];rendered=self.render(prompt,len(images))
  return rendered,images,verified_label_ids(self.processor.tokenizer,rendered,labels)
 def collate(self,examples):
  """examples: (rendered, images, token_ids, target). Returns model inputs plus per-example label ids and targets."""
  self.processor.tokenizer.padding_side='left'
  image_groups=[e[1] for e in examples]
  images=None if all(not group for group in image_groups) else image_groups
  inputs=self.processor(text=[e[0] for e in examples],images=images,return_tensors='pt',padding=True,**({'pad_to_multiple_of':self.pad_multiple} if self.pad_multiple else {}))
  if inputs['input_ids'].shape[-1]>self.max_length:raise ValueError(f'Processed request exceeds the {self.max_length}-token limit')
  suffix=self.processor.tokenizer.encode('</think>\n\n',add_special_tokens=False)
  if not all(row[-len(suffix):].tolist()==suffix for row in inputs['input_ids']):raise ValueError('Left padding did not align the decision positions')
  return inputs,[e[2] for e in examples],[e[3] for e in examples]
 def candidate_logits_batch(self,inputs,token_ids):
  inputs={k:v.to(self.device) for k,v in inputs.items()};base=self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
  hidden=base.model(**inputs).last_hidden_state[:,-1].float();head=base.lm_head.weight
  result=[]
  for i,ids in enumerate(token_ids):
   indices=self._readout_indices(ids)
   result.append(self.readout(hidden[i])[indices] if self.readout is not None and indices is not None else hidden[i]@head[torch.tensor(ids,device=self.device)].float().T)
  return result
