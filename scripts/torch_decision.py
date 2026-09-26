"""PyTorch twin of the MLX decision path, for cloud GPUs. Same prompt, same decision position, same float32 candidate head."""
from pathlib import Path
import json
import torch
from vision_decision.scoring import MAX_READOUT_CODES, DEFAULT_PROMPT_LAYOUT, check_prompt_layout, check_readout_codes, readout_codes, verified_label_ids

TARGETS=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','in_proj_qkv','in_proj_z','out_proj']

class TorchDecision:
 def __init__(self,path,device,dtype=torch.bfloat16,max_length=4096,pad_multiple=0):
  from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
  self.processor=AutoProcessor.from_pretrained(path,local_files_only=True);self.device=device;self.max_length=max_length;self.pad_multiple=pad_multiple  # pad batches to a multiple (fewer distinct kernel shapes)
  self.model=Qwen3_5ForConditionalGeneration.from_pretrained(path,local_files_only=True,dtype=dtype).to(device).eval()
  self.readout=None;self._codebook=None
  self.codes=MAX_READOUT_CODES  # readout size: 255 shipped; 256 with the extended readout (enable_readout(codes=256))
  self.prompt_layout=DEFAULT_PROMPT_LAYOUT  # set from the adapter's decision_readout.json, or by the trainer
 def add_lora(self,rank=16,alpha=32,targets=None):
  from peft import LoraConfig,get_peft_model
  # Language layers only, as in the MLX run; the vision tower stays frozen. `targets` narrows the module list (e.g. attention only).
  self.model=get_peft_model(self.model,LoraConfig(r=rank,lora_alpha=alpha,lora_dropout=0.0,target_modules=r'.*language_model.*\.('+'|'.join(targets or TARGETS)+')'))
  return self.model
 def _ensure_codebook(self,n_images=0):
  if self._codebook is None or len(self._codebook)!=self.codes:self._codebook=readout_codes(self.processor.tokenizer,self.render('',n_images),self.codes,limit=self.codes)
  return self._codebook
 def labels(self,count,n_images=0):
  codebook=self._ensure_codebook(n_images)
  if count>len(codebook):raise ValueError(f'{count} candidates exceed the {len(codebook)}-code readout (at most {len(codebook)-1} options + unknown)')
  return [x[0] for x in codebook[:count]]
 @property
 def max_options(self):return self.codes-1
 def enable_readout(self,adapter=None,trainable=True,codes=None):
  """Install the decision readout Linear[codes, hidden], initialized from the matching LM-head rows.

  codes: 255 (shipped) or 256 (extended); None = the adapter's own row count (255 without an adapter).
  A 255-row trained readout loaded with codes=256 gets row 256 appended from its LM-head row, exactly as
  the original rows were initialized, so questions with <= 254 options use unchanged rows.
  The adapter's prompt layout (decision_readout.json "prompt_layout", absent = standard) is adopted.
  Older adapters omit decision_readout.safetensors and continue through vocabulary rows.
  """
  from safetensors.torch import load_file
  base=self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
  trained=manifest=None
  if adapter is not None:
   path=Path(adapter)/'decision_readout.safetensors'
   if not path.exists():return False
   trained=load_file(str(path),device=str(self.device))['weight'].float()
   manifest_path=Path(adapter)/'decision_readout.json'
   if not manifest_path.exists():raise ValueError('Trained readout is missing decision_readout.json tokenizer binding')
   manifest=json.loads(manifest_path.read_text())
  rows=trained.shape[0] if trained is not None and trained.ndim==2 else None
  wanted=check_readout_codes(codes if codes is not None else rows if rows in (255,256) else MAX_READOUT_CODES)
  if trained is not None and rows not in (255,256):raise ValueError('Decision readout must be finite with shape [255 or 256, hidden_size]')
  if rows is not None and rows>wanted:raise ValueError(f'This adapter has a {rows}-code readout; load it with codes={rows}')
  self.codes=wanted;self._codebook=None;codebook=self._ensure_codebook(0)
  ids=[x[1] for x in codebook]
  weight=base.lm_head.weight[torch.tensor(ids,device=self.device)].detach().float().clone()
  if trained is not None:
   actual=[{'code':c,'token_id':i} for c,i in codebook]
   bound=manifest.get('codes')
   if manifest.get('version')!=1 or not isinstance(bound,list) or len(bound)!=rows or bound!=actual[:rows]:raise ValueError('Decision readout code/token binding does not match this tokenizer')
   weight=torch.cat([trained,weight[rows:]]) if rows<wanted else trained  # appended rows: LM-head rows, as at initialization
   self.prompt_layout=check_prompt_layout(manifest.get('prompt_layout',DEFAULT_PROMPT_LAYOUT))
  if tuple(weight.shape)!=(self.codes,base.lm_head.weight.shape[1]) or not bool(torch.isfinite(weight).all()):raise ValueError(f'Decision readout must be finite with shape [{self.codes}, hidden_size]')
  self.readout=torch.nn.Linear(weight.shape[1],self.codes,bias=False,device=self.device,dtype=torch.float32)
  self.readout.weight.data.copy_(weight);self.readout.weight.requires_grad_(trainable)
  return True
 def save_readout(self,directory):
  if self.readout is None:raise ValueError('Decision readout is not enabled')
  from safetensors.torch import save_file
  save_file({'weight':self.readout.weight.detach().cpu().contiguous()},str(Path(directory)/'decision_readout.safetensors'))
  manifest={'version':1,'codes':[{'code':c,'token_id':i} for c,i in self._codebook]}
  if self.prompt_layout!=DEFAULT_PROMPT_LAYOUT:manifest['prompt_layout']=check_prompt_layout(self.prompt_layout)  # absent = standard, so shipped adapters stay byte-identical
  (Path(directory)/'decision_readout.json').write_text(json.dumps(manifest,indent=2)+'\n')
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
 def candidate_logits_with_rationale(self,inputs,token_ids,position,labels):
  """--rationale-weight path: candidate logits read at `position` (the decision position, unchanged by the
  appended rationale block under causal attention) plus the per-example rationale LM loss [B]."""
  from decision_recipe import rationale_lm_loss
  inputs={k:v.to(self.device) for k,v in inputs.items()};base=self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
  states=base.model(**inputs).last_hidden_state;hidden=states[:,position].float();head=base.lm_head.weight
  result=[]
  for i,ids in enumerate(token_ids):
   indices=self._readout_indices(ids)
   result.append(self.readout(hidden[i])[indices] if self.readout is not None and indices is not None else hidden[i]@head[torch.tensor(ids,device=self.device)].float().T)
  return result,rationale_lm_loss(states,head,labels,position+1)
