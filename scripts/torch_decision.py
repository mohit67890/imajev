"""PyTorch twin of the MLX decision path, for cloud GPUs. Same prompt, same decision position, same float32 candidate head."""
from pathlib import Path
import json
import torch
from vision_decision.scoring import MAX_READOUT_CODES, DEFAULT_PROMPT_LAYOUT, check_prompt_layout, check_readout_codes, readout_codes, verified_label_ids

TARGETS=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','in_proj_qkv','in_proj_z','out_proj']

class _LazyMultimodalProcessor:
 """Text requests need only the tokenizer; load optional vision/video processors on demand."""
 def __init__(self,path):
  from transformers import AutoTokenizer
  self.path=path;self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True);self._multimodal=None
 def _vision(self):
  if self._multimodal is None:
   from transformers import AutoProcessor
   self._multimodal=AutoProcessor.from_pretrained(self.path,local_files_only=True)
   self._multimodal.tokenizer=self.tokenizer  # preserve padding-side changes made by collate()
  return self._multimodal
 def apply_chat_template(self,messages,**kwargs):
  if any(item.get('type') in ('image','video') for message in messages for item in message.get('content',[]) if isinstance(item,dict)):
   return self._vision().apply_chat_template(messages,**kwargs)
  return self.tokenizer.apply_chat_template(messages,**kwargs)
 def __call__(self,text,images=None,**kwargs):
  if images is not None:return self._vision()(text=text,images=images,**kwargs)
  kwargs.setdefault('add_special_tokens',False)
  return self.tokenizer(text,**kwargs)
 def __getattr__(self,name):
  if name.startswith('_'):raise AttributeError(name)
  return getattr(self._vision(),name)

class TorchDecision:
 def __init__(self,path,device,dtype=torch.bfloat16,max_length=4096,pad_multiple=0):
  from transformers import Qwen3_5ForConditionalGeneration
  self.processor=_LazyMultimodalProcessor(path);self.device=device;self.max_length=max_length;self.pad_multiple=pad_multiple  # pad batches to a multiple (fewer distinct kernel shapes)
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
 def collate_fast(self,examples):
  """collate() with fast-path pixel handling: uint8 passthrough (no CPU
  rescale/normalize); normalize_patches finishes on GPU in candidate_logits_fast."""
  self.processor.tokenizer.padding_side='left'
  image_groups=[e[1] for e in examples]
  images=None if all(not group for group in image_groups) else image_groups
  inputs=dict(self.processor(text=[e[0] for e in examples],images=images,return_tensors='pt',do_rescale=False,do_normalize=False,padding=True,**({'pad_to_multiple_of':self.pad_multiple} if self.pad_multiple else {})))
  if inputs['input_ids'].shape[-1]>self.max_length:raise ValueError(f'Processed request exceeds the {self.max_length}-token limit')
  suffix=self.__dict__.get('_suffix_ids') or self.__dict__.setdefault('_suffix_ids',self.processor.tokenizer.encode(self.DECISION_TAIL,add_special_tokens=False))
  if not all(row[-len(suffix):].tolist()==suffix for row in inputs['input_ids']):raise ValueError('Left padding did not align the decision positions')
  return inputs,[e[2] for e in examples],[e[3] for e in examples]
 def candidate_logits_batch_fast(self,inputs,token_ids):
  """candidate_logits_batch() through the fast forward. Graphs are captured at
  batch 1, so replay is gated on B==1; real batches run the eager LM (keeps the
  uint8 + single-tokenize fast-path wins either way)."""
  inputs={k:v.to(self.device) for k,v in inputs.items()}
  if inputs.get('pixel_values') is not None and inputs['pixel_values'].dtype==torch.uint8:inputs['pixel_values']=self.normalize_patches(inputs['pixel_values'])
  embeds,positions=self._embeds_positions(inputs);graphs=self.__dict__.get('graphs')
  # Graphs are captured at batch 1; replay with B>1 would silently score row 0.
  if graphs is not None and embeds.shape[0] == 1 and graphs.fits(embeds.shape[1]):
   hidden=graphs.run(embeds,positions).float().unsqueeze(0)
   base=self._base();head=base.lm_head.weight
   return [self.readout(hidden[i])[self._readout_indices(ids)] if self.readout is not None and self._readout_indices(ids) is not None else hidden[i]@head[torch.tensor(ids,device=self.device)].float().T for i,ids in enumerate(token_ids)]
  base=self._base()
  hidden=base.model.language_model(inputs_embeds=embeds,position_ids=positions,use_cache=False).last_hidden_state[:,-1].float();head=base.lm_head.weight
  return [self.readout(hidden[i])[self._readout_indices(ids)] if self.readout is not None and self._readout_indices(ids) is not None else hidden[i]@head[torch.tensor(ids,device=self.device)].float().T for i,ids in enumerate(token_ids)]
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

 # ---- serving fast path (server --fast): same decision function, less work per request ----
 def merge_adapter(self,adapter,dtype=torch.bfloat16):
  """Fold a PEFT LoRA into the weights so every forward is a plain base-model forward. Call on a float32 model: the
  low-rank sum is formed in float32 and rounded once to `dtype`. The decision readout is separate and stays float32."""
  from peft import PeftModel
  self.model=PeftModel.from_pretrained(self.model,str(adapter)).merge_and_unload().to(dtype).eval()
 DECISION_TAIL='</think>\n\n'
 def label_ids(self,rendered,labels):
  """= verified_label_ids(tokenizer, rendered, labels), with one tokenization per label for the life of the process.
  The rendered prompt ends with the special token '</think>' then '\\n\\n'; the tokenizer splits at special tokens, so
  whether a label is one token at the decision position depends only on that tail (test_torch_fast_path checks this)."""
  if not rendered.endswith(self.DECISION_TAIL):return verified_label_ids(self.processor.tokenizer,rendered,labels)
  cache=self.__dict__.setdefault('_label_cache',{})
  for label in labels:
   if label not in cache:cache[label]=verified_label_ids(self.processor.tokenizer,self.DECISION_TAIL,[label])[0]
  ids=[cache[label] for label in labels]
  if len(set(ids))!=len(ids):raise ValueError('Choice labels do not have distinct token IDs')
  return ids
 def prepare_fast(self,images,prompt,labels):
  """prepare() with one tokenization: text-only requests skip the multimodal processor (same ids, checked by the test)."""
  images=[] if images is None else images if isinstance(images,list) else [images];rendered=self.render(prompt,len(images))
  token_ids=self.label_ids(rendered,labels)
  # Images: the processor resizes and patchifies as always but leaves the pixels as uint8; normalize_patches finishes the
  # identical float32 arithmetic on the GPU (8x fewer bytes to copy, no float work on the CPU).
  inputs=dict(self.processor(text=[rendered],images=images,return_tensors='pt',do_rescale=False,do_normalize=False)) if images else {'input_ids':self.processor.tokenizer(rendered,add_special_tokens=False,return_tensors='pt')['input_ids']}
  if inputs['input_ids'].shape[-1]>self.max_length:raise ValueError(f'Processed request exceeds the {self.max_length}-token limit')
  suffix=self.__dict__.get('_suffix_ids') or self.__dict__.setdefault('_suffix_ids',self.processor.tokenizer.encode(self.DECISION_TAIL,add_special_tokens=False))
  if inputs['input_ids'][0,-len(suffix):].tolist()!=suffix:raise ValueError('Processed decision-position suffix mismatch')
  return rendered,inputs,token_ids
 def normalize_patches(self,patches):
  """The image processor's rescale_and_normalize on already patchified uint8 pixels: one fused mean and std per channel,
  then x.float() - mean, / std, exactly the torchvision normalize the processor runs before patchify. Elementwise, so
  patchify first changes nothing; each patch row is channel-major (channel x temporal x patch x patch)."""
  ip=self.processor.image_processor
  mean,std,_=ip._fuse_mean_std_and_rescale_factor(do_normalize=True,image_mean=ip.image_mean,image_std=ip.image_std,do_rescale=True,rescale_factor=ip.rescale_factor,device=patches.device)
  per=ip.temporal_patch_size*ip.patch_size*ip.patch_size
  mean=torch.as_tensor(mean,dtype=torch.float32,device=patches.device).repeat_interleave(per);std=torch.as_tensor(std,dtype=torch.float32,device=patches.device).repeat_interleave(per)
  return patches.to(torch.float32).sub_(mean).div_(std)
 def _base(self):
  """The Qwen3.5 model under a PEFT wrapper (its LoRA layers stay in place) or the merged model itself."""
  return self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
 def _embeds_positions(self,inputs):
  """The multimodal model's forward up to the language model: token embeddings with the image features scattered in, and the
  M-RoPE position ids (3 x 1 x n). Text-only positions are 0..n-1 on every axis, as the model's own default."""
  m=self._base().model;ids=inputs['input_ids'];embeds=m.get_input_embeddings()(ids)
  if 'pixel_values' not in inputs:return embeds,torch.arange(ids.shape[1],device=ids.device).view(1,1,-1).expand(3,1,-1)
  features=torch.cat(m.get_image_features(inputs['pixel_values'],inputs['image_grid_thw'],return_dict=True).pooler_output,dim=0).to(embeds.device,embeds.dtype)
  mask,_=m.get_placeholder_mask(ids,inputs_embeds=embeds,image_features=features);embeds=embeds.masked_scatter(mask,features)
  positions=m.compute_3d_position_ids(input_ids=ids,inputs_embeds=embeds,image_grid_thw=inputs['image_grid_thw'],attention_mask=inputs.get('attention_mask'),mm_token_type_ids=inputs.get('mm_token_type_ids'))
  return embeds,positions
 def candidate_logits_fast(self,inputs,token_ids):
  """candidate_logits(), through a recorded CUDA graph when one fits the prompt length."""
  inputs={k:v.to(self.device) for k,v in inputs.items()};base=self._base()
  if inputs.get('pixel_values') is not None and inputs['pixel_values'].dtype==torch.uint8:inputs['pixel_values']=self.normalize_patches(inputs['pixel_values'])
  embeds,positions=self._embeds_positions(inputs);graphs=self.__dict__.get('graphs')
  hidden=graphs.run(embeds,positions) if graphs is not None and graphs.fits(embeds.shape[1]) else base.model.language_model(inputs_embeds=embeds,position_ids=positions,use_cache=False).last_hidden_state[0,-1]
  indices=self._readout_indices(token_ids)
  return self.readout(hidden.float())[indices] if self.readout is not None and indices is not None else hidden.float()@base.lm_head.weight[torch.tensor(token_ids,device=self.device)].float().T
 def capture_graphs(self,lengths):
  base=self._base()
  self.graphs=DecoderGraphs(base.model.language_model,base.config.text_config.hidden_size,lengths,self.device,base.dtype)
  return self.graphs

 # ---- thinking (think-if-unsure): a greedy thought in Qwen3.5's thinking template, the decision read right after it ----
 THINK_TAIL='<think>\n'
 def render_thinking(self,prompt,n_images):
  messages=[dict(role='user',content=[dict(type='image')]*n_images+[dict(type='text',text=prompt)])]
  rendered=self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,enable_thinking=True)
  if not rendered.endswith(self.THINK_TAIL):raise ValueError('Unexpected Qwen thinking template boundary')
  return rendered
 def _thinking_inputs(self,images,prompt):
  images=[] if images is None else images if isinstance(images,list) else [images]
  inputs=dict(self.processor(text=[self.render_thinking(prompt,len(images))],images=images or None,return_tensors='pt'))
  if inputs['input_ids'].shape[-1]>self.max_length:raise ValueError(f'Processed request exceeds the {self.max_length}-token limit')
  return inputs
 def think_end_id(self):
  ids=self.processor.tokenizer.encode('</think>',add_special_tokens=False)
  if len(ids)!=1:raise ValueError("'</think>' is not a single token")
  return ids[0]
 @torch.inference_mode()
 def generate_thought(self,images,prompt,max_tokens):
  """Greedy thought token ids (without '</think>') and whether the model closed the thought itself."""
  tok=self.processor.tokenizer;end=self.think_end_id();pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
  inputs={k:v.to(self.device) for k,v in self._thinking_inputs(images,prompt).items()}
  out=self.model.generate(**inputs,max_new_tokens=int(max_tokens),do_sample=False,eos_token_id=[end,tok.eos_token_id],pad_token_id=pad)
  tokens=[]
  for token in out[0,inputs['input_ids'].shape[-1]:].tolist():
   if token==end:return tokens,True
   if token in (tok.eos_token_id,pad):break
   tokens.append(token)
  return tokens,False
 def inputs_after_thought(self,images,prompt,labels,thought,closed):
  """Model inputs ending '</think>\\n\\n' after `thought` (token ids; a cut thought is closed with '\\n</think>\\n\\n'),
  and the labels' token ids at that decision position. Same image, prompt and decision position as the single pass."""
  images=[] if images is None else images if isinstance(images,list) else [images]
  inputs=self._thinking_inputs(images,prompt);tok=self.processor.tokenizer
  tail=tok.encode('</think>\n\n' if closed else '\n</think>\n\n',add_special_tokens=False)
  extra=torch.tensor([list(thought)+tail],dtype=inputs['input_ids'].dtype)
  inputs['input_ids']=torch.cat([inputs['input_ids'],extra],dim=1)
  if 'attention_mask' in inputs:inputs['attention_mask']=torch.cat([inputs['attention_mask'],torch.ones_like(extra)],dim=1)
  if 'mm_token_type_ids' in inputs:inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(extra,dtype=inputs['mm_token_type_ids'].dtype)],dim=1)
  if inputs['input_ids'].shape[-1]>self.max_length:raise ValueError(f'Processed request exceeds the {self.max_length}-token limit')
  suffix=tok.encode(self.DECISION_TAIL,add_special_tokens=False)
  if inputs['input_ids'][0,-len(suffix):].tolist()!=suffix:raise ValueError('Decision-position suffix mismatch after the thought')
  return inputs,self.label_ids(self.DECISION_TAIL,labels)

GRAPH_LENGTHS=list(range(64,1025,64))+list(range(1152,2049,128))+list(range(2304,4097,256))

class DecoderGraphs:
 """CUDA graphs of the language model at fixed padded lengths, recorded once at load (the design of JevK5's runtime,
 Apache-2.0, also used by FlyMy's Decision 4B). A prompt is right-padded to the next recorded length: every layer is
 causal (full attention, gated DeltaNet, its short convolution), so padding after the decision position cannot change it."""
 def __init__(self,lm,hidden,lengths,device,dtype):
  self.graphs={};pool=None
  with torch.inference_mode():
   for n in sorted(set(lengths),reverse=True):  # longest first: the shorter graphs reuse its memory pool
    embeds=torch.zeros(1,n,hidden,device=device,dtype=dtype);positions=torch.arange(n,device=device).view(1,1,n).expand(3,1,n).contiguous()
    last=torch.zeros(1,dtype=torch.long,device=device)
    def step(embeds=embeds,positions=positions,last=last):
     return lm(inputs_embeds=embeds,position_ids=positions,use_cache=False).last_hidden_state[0].index_select(0,last)[0]
    side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
     for _ in range(3):step()  # warm-up: compiles the Triton kernels outside the capture
    torch.cuda.current_stream().wait_stream(side)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,pool=pool):out=step()
    pool=graph.pool();self.graphs[n]=(graph,embeds,positions,last,out)
  self.lengths=sorted(self.graphs)
 def fits(self,n):return bool(self.lengths) and n<=self.lengths[-1]
 def run(self,embeds,positions):
  n=embeds.shape[1];size=next(x for x in self.lengths if x>=n);graph,static_embeds,static_positions,last,out=self.graphs[size]
  static_embeds.zero_();static_embeds[:,:n].copy_(embeds)
  static_positions[:,:,:n].copy_(positions);static_positions[:,:,n:].copy_(positions[:,:,-1:]+torch.arange(1,size-n+1,device=positions.device))
  last.fill_(n-1);graph.replay()
  return out.clone()
