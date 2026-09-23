"""PyTorch/PEFT twin of train_decision_lora.py for cloud GPUs. Same data, prompt, loss and schedule.

Multi-GPU: launch with `torchrun --nproc_per_node=N`. Each rank takes every N-th example of the same
deterministic stream and LoRA gradients are averaged once per optimizer step, so N GPUs with
--accumulate A see exactly the examples one GPU would see with --accumulate A*N.

Preemption-safe: checkpoints (adapter + optimizer + step) go to --output, which may be a mounted
bucket path; rerunning the same command resumes. --model may be a local snapshot or a Hub repo id.
"""
import gc,os,json,math,time,random,argparse,hashlib,shutil
from pathlib import Path
import torch
import torch.distributed as dist
import faulthandler,sys
None  # periodic stack dumps disabled: they coincided with the SIGSEGVs (the dump thread walks frames without the GIL)  # a stack every 4 minutes of silence: a stalled run explains itself
from decision_data import load_records,load_image,render,pixel_budget,batch_plan,with_noise_state,approx_tokens
from torch_decision import TorchDecision,TARGETS

parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--model',required=True);parser.add_argument('--revision')
parser.add_argument('--device',default='cuda');parser.add_argument('--epochs',type=float,default=1);parser.add_argument('--lr',type=float,default=5e-5);parser.add_argument('--warmup',type=int,default=20)
parser.add_argument('--init-adapter',help='v1 PEFT adapter used to initialize the LoRA weights (the v1.1 readout is initialized from LM rows)')
parser.add_argument('--accumulate',type=int,default=8);parser.add_argument('--batch-size',type=int,default=1,help='maximum examples per micro-batch');parser.add_argument('--token-budget',type=int,default=0,help='padded tokens per micro-batch (longest x count); 0 = fixed batch size, no bucketing');parser.add_argument('--workers',type=int,default=0);parser.add_argument('--no-checkpointing',action='store_true');parser.add_argument('--max-steps',type=int,default=0,help='stop after N optimizer steps (throughput probes)');parser.add_argument('--clip',type=float,default=1.0);parser.add_argument('--rank',type=int,default=16);parser.add_argument('--alpha',type=float,default=32)
parser.add_argument('--max-length',type=int,default=4096,help='refuse processed examples beyond this token length; never truncate');parser.add_argument('--lora-targets',default='',help='comma-separated LoRA target module names (default: all language-layer projections); ignored when --init-adapter is given (its structure is kept)');parser.add_argument('--dev2-version',default='',help='second dev manifest (e.g. a reasoning-style set) evaluated at every dev checkpoint');parser.add_argument('--dev2-cases',type=int,default=400);parser.add_argument('--select',choices=['dev_loss','dev2_accuracy','mean_accuracy'],default='dev_loss',help='what "best" checkpoint means');parser.add_argument('--pad-multiple',type=int,default=0,help='pad each micro-batch to a multiple of this many tokens (0 = exact); fewer kernel shapes for the JIT kernels')
parser.add_argument('--pixels',type=int,default=400000);parser.add_argument('--dev-every',type=int,default=50);parser.add_argument('--save-every',type=int,default=0,help='also save a resumable checkpoint every N steps without a dev pass (0 = only at dev checkpoints)');parser.add_argument('--dev-cases',type=int,default=200);parser.add_argument('--limit',type=int,default=0)
parser.add_argument('--seed',type=int,default=0);parser.add_argument('--version',default='decision-v0');parser.add_argument('--max-hours',type=float,default=0,help='stop cleanly after this many hours (cost cap)');args=parser.parse_args()
world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));main=rank==0
if world>1:
 dist.init_process_group('nccl' if args.device=='cuda' else 'gloo')
 if args.device=='cuda':torch.cuda.set_device(int(os.environ['LOCAL_RANK']));args.device=f"cuda:{os.environ['LOCAL_RANK']}"
def say(*a):
 if main:print(*a,flush=True)
def across(*values):  # sum python numbers across ranks
 t=torch.tensor(values,dtype=torch.float64,device=args.device if args.device!='mps' else 'cpu')
 if world>1:dist.all_reduce(t)
 return t.tolist()
started=time.monotonic();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);VERSION=args.version;train=load_records('train',VERSION);dev=load_records('dev',VERSION);dev=dev[::max(1,len(dev)//args.dev_cases)][:args.dev_cases]
dev2=load_records('dev',args.dev2_version) if args.dev2_version else [];dev2=dev2[::max(1,len(dev2)//args.dev2_cases)][:args.dev2_cases] if dev2 else []
if args.limit:train=train[:args.limit]
path=args.model
if not Path(path).is_dir():
 from huggingface_hub import snapshot_download
 path=snapshot_download(args.model,revision=args.revision)
config=dict({k:v for k,v in vars(args).items() if k not in ('output','device','max_hours','workers','max_steps')},world_size=world,manifest_sha256=hashlib.sha256(Path(f'data/manifests/{VERSION}.jsonl').read_bytes()).hexdigest(),train_cases=len(train),dev_cases=len(dev),targets=TARGETS,
 loss='cross-entropy over float32 candidate-label logits at the served decision position; no label smoothing')
p=out/'config.json'
VOLATILE={'dev_every','dev_cases','save_every'}  # checkpoint cadence may change on resume; everything that affects the data order or the optimisation may not
strip=lambda c:{k:v for k,v in c.items() if k not in VOLATILE}
if p.exists():assert strip(json.loads(p.read_text()))==strip(config),'Output exists with a different config'
elif main:p.write_text(json.dumps(config,indent=2)+'\n')
torch.manual_seed(args.seed);engine=TorchDecision(path,args.device,max_length=args.max_length,pad_multiple=args.pad_multiple);model=engine.add_lora(args.rank,args.alpha,[t for t in args.lora_targets.split(',') if t] or None)
init=None
if args.init_adapter:
 from peft import set_peft_model_state_dict
 from safetensors.torch import load_file
 init=Path(args.init_adapter);set_peft_model_state_dict(model,load_file(str(init/'adapter_model.safetensors')))
engine.enable_readout(args.init_adapter if init is not None and (init/'decision_readout.safetensors').exists() else None,trainable=True);model.train();engine.readout.train()
if args.device.startswith('cuda') and not args.no_checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False));model.enable_input_require_grads()
params=[q for q in model.parameters() if q.requires_grad]+list(engine.readout.parameters());# One deterministic plan of length-bucketed micro-batches per epoch, identical on every rank; rank r takes every world-th batch.
budget=args.token_budget or 10**9
def plan(epoch):
 return batch_plan(train,args.pixels,budget,args.batch_size,args.seed,epoch)
epochs_needed=math.ceil(args.epochs);plans=[plan(e) for e in range(epochs_needed)];flat=[(e,b) for e in range(epochs_needed) for b in plans[e]]
flat=flat[:round(len(flat)*args.epochs/epochs_needed)];per_step=args.accumulate*world;original_microbatches=len(flat);steps=math.ceil(len(flat)/per_step)
# A distributed optimizer step must have the same accumulation count on every rank. Repeat a
# deterministic prefix only to fill the final step; every planned example is still consumed.
needed=steps*per_step-len(flat)
if needed:flat += [flat[i%len(flat)] for i in range(needed)]
# Balance each optimizer step across ranks: sort the step's batches by padded cost and deal them out in snake order,
# so no rank ends up with all the heavy batches while the others wait at the gradient sync.
def cost(item):
 epoch,batch=item;return max(approx_tokens(train[j],args.pixels) for j in batch)*len(batch)
balanced=[]
for s0 in range(0,steps*per_step,per_step):
 group=sorted(flat[s0:s0+per_step],key=cost,reverse=True);slots=[[] for _ in range(world)]
 for k,item in enumerate(group):
  r=k%world if (k//world)%2==0 else world-1-(k%world);slots[r].append(item)
 for a in range(args.accumulate):
  for r in range(world):balanced.append(slots[r][a])
flat=balanced
if world>1:
 for q in params:dist.broadcast(q.data,0)
optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.0)
def rate(step):  # linear warm-up, cosine decay to 10%, as in the MLX run
 if step<args.warmup:return (step+1)/args.warmup
 return .1+.9*.5*(1+math.cos(math.pi*(step-args.warmup)/max(1,steps-args.warmup)))
scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
def make(record,rng):
 if rng is not None:record=with_noise_state(record,rng)
 header,choices,texts,target=render(record,rng);labels=engine.labels(len(choices),len(record['images']))
 images=[load_image(x['image'],pixel_budget(record,args.pixels)) for x in record['images']]
 return (*engine.render_example(images,header+'\n'.join(f'{l}: {t}' for l,t in zip(labels,texts)),labels),target)
def collate_within_budget(examples):
 """Collate a planned micro-batch; if its real padded size overshoots the plan (the estimate is approximate, and
 mixed text/image batches overshoot most), split it in halves. Every example is still trained on."""
 inputs,ids,targets=engine.collate(examples);length=inputs['input_ids'].shape[-1]
 limit=budget if length<=2048 else budget*0.6  # full-attention layers grow quadratically with sequence length
 if len(examples)>1 and length*len(examples)>limit:
  half=len(examples)//2;return collate_within_budget(examples[:half])+collate_within_budget(examples[half:])
 return [((inputs,ids,targets),examples)]
class Stream(torch.utils.data.Dataset):  # item = one planned micro-batch for this rank as a list of collated chunks; deterministic, so a resumed run continues the same sequence
 def __init__(self,first_step):self.items=[flat[i] for i in range(first_step*per_step,steps*per_step) if i%world==rank]
 def __len__(self):return len(self.items)
 def __getitem__(self,i):
  epoch,batch=self.items[i];return collate_within_budget([make(train[j],random.Random(f'{args.seed}:{epoch}:{j}')) for j in batch])
class Dev(torch.utils.data.Dataset):
 def __init__(self,rows=None):self.rows=dev if rows is None else rows
 def __len__(self):return len(range(rank,len(self.rows),world))
 def __getitem__(self,i):return make(self.rows[rank+i*world],None)
def loader(dataset,collated):
 extra=dict(batch_size=None) if collated else dict(batch_size=min(args.batch_size,8),collate_fn=engine.collate)
 return torch.utils.data.DataLoader(dataset,shuffle=False,num_workers=args.workers,prefetch_factor=4 if args.workers else None,**extra)
def batch_loss(batch):
 inputs,token_ids,targets=batch;logits=engine.candidate_logits_batch(inputs,token_ids)
 def one(x,t):  # hard label, or a soft distribution over the listed candidates
  if isinstance(t,list):return -(torch.tensor(t,device=x.device,dtype=x.dtype)*x.log_softmax(0)).sum()
  return torch.nn.functional.cross_entropy(x[None],torch.tensor([t],device=x.device))[None].squeeze()
 mode=lambda t:max(range(len(t)),key=t.__getitem__) if isinstance(t,list) else t
 return torch.stack([one(x,t) for x,t in zip(logits,targets)]).sum(),sum(int(x.argmax())==mode(t) for x,t in zip(logits,targets)),len(targets)
def dev_loss(rows=None):
 model.eval();total=0;hits=0;n=0
 with torch.no_grad():
  for batch in loader(Dev(rows),False):
   try:loss,hit,size=batch_loss(batch);total+=float(loss);hits+=hit;n+=size
   except torch.OutOfMemoryError:
    torch.cuda.empty_cache();raise RuntimeError('Development batch exceeded memory; lower --token-budget or --batch-size (evaluation may not skip records)')
 model.train();total,hits,n=across(total,hits,n);return total/n,hits/n
def evaluate_dev(entry):
 """Fill dev metrics into the log entry and return the selection score (lower is better)."""
 entry['dev_loss'],entry['dev_accuracy']=dev_loss()
 if dev2:entry['dev2_loss'],entry['dev2_accuracy']=dev_loss(dev2)
 if args.select=='dev2_accuracy':return -entry.get('dev2_accuracy',entry['dev_accuracy'])
 if args.select=='mean_accuracy':return -(entry['dev_accuracy']+entry.get('dev2_accuracy',entry['dev_accuracy']))/2
 return entry['dev_loss']
def save(name,step,best):
 if world>1:dist.barrier()
 if not main:return
 tmp=out/(name+'.tmp');shutil.rmtree(tmp,ignore_errors=True);model.save_pretrained(tmp)
 engine.save_readout(tmp)
 torch.save(dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,best=best),tmp/'trainer.pt')
 shutil.rmtree(out/name,ignore_errors=True);tmp.rename(out/name)
log=out/'log.jsonl';step0=0;best=None
if (out/'last'/'trainer.pt').exists():
 from peft import set_peft_model_state_dict;from safetensors.torch import load_file
 set_peft_model_state_dict(model,load_file(str(out/'last'/'adapter_model.safetensors')));state=torch.load(out/'last'/'trainer.pt',map_location=args.device)
 engine.readout.weight.data.copy_(load_file(str(out/'last'/'decision_readout.safetensors'),device=str(args.device))['weight'])
 optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);step0,best=state['step'],state['best'];say('Resumed at step',step0)
else:
 entry=dict(step=0);best=evaluate_dev(entry);say(json.dumps(entry))
 save('best',0,best)
 if main:
  with log.open('a') as f:f.write(json.dumps(entry)+'\n')
stream=iter(loader(Stream(step0),True));seen=0;skipped=0;window=time.monotonic()
skip_file=out/'skip_steps.txt'
if skip_file.exists() and skip_file.read_text().split():raise ValueError('skip_steps.txt is no longer supported: v1.1 training may not skip data')
for step in range(step0,steps):
 start=time.monotonic();losses=[];counted=0;failed_oom=0;optimizer.zero_grad(set_to_none=True)
 for k in range(args.accumulate):
  try:batch=next(stream)
  except StopIteration:break
  def train_chunk(collated,examples):  # on out-of-memory, halve the chunk and retry rather than skip; a single example that does not fit is a hard error
   global seen,counted
   oom=False
   try:
    loss,_,size=batch_loss(collated);loss.backward();losses.append(float(loss.detach()));seen+=size;counted+=size
   except torch.OutOfMemoryError:
    oom=True  # retry outside the handler: the exception's traceback would otherwise keep the failed step's activations alive
   if not oom:return
   collated=None;gc.collect();torch.cuda.empty_cache()
   if len(examples)<2:raise torch.OutOfMemoryError(f'step {step+1}: a single example does not fit in memory')
   half=len(examples)//2;say(f'step {step+1}: chunk of {len(examples)} exceeded memory, retrying as {half}+{len(examples)-half}')
   for part in (examples[:half],examples[half:]):train_chunk(engine.collate(part),part)
  try:
   for collated,examples in batch:train_chunk(collated,examples)
  except torch.OutOfMemoryError:  # a single example did not fit; every collective below is still reached, so ranks stay in step
   skipped+=1;failed_oom=1;torch.cuda.empty_cache();break
 if world>1:
  failure=torch.tensor([failed_oom],device=args.device,dtype=torch.int32);dist.all_reduce(failure,op=dist.ReduceOp.MAX);failed_oom=int(failure.item())
 if failed_oom:raise RuntimeError(f'Training batch exceeded memory at step {step+1}; lower --token-budget (data skipping is disabled)')
 if world>1:  # one all-reduce of the (small) LoRA gradient per optimizer step
  flat_grad=torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).reshape(-1) for q in params]);dist.all_reduce(flat_grad);offset=0
  for q in params:q.grad=flat_grad[offset:offset+q.numel()].view_as(q).clone();offset+=q.numel()
 examples=max(1.0,across(counted)[0])  # gradients were summed over examples on every rank: turn them into a per-example mean
 for q in params:
  if q.grad is not None:q.grad/=examples
 norm=float(torch.nn.utils.clip_grad_norm_(params,args.clip));optimizer.step();scheduler.step()
 loss_sum,seen_all,skipped_all=across(sum(losses),seen,skipped);count=examples;entry=dict(examples_in_step=int(examples),skipped_oom_batches=int(skipped_all),step=step+1,of=steps,loss=loss_sum/count,grad_norm=norm,seconds=time.monotonic()-start,examples_per_second=seen_all/(time.monotonic()-window),gpus=world);assert math.isfinite(entry['loss']) and math.isfinite(norm)
 out_of_time=bool(across(float(bool(args.max_hours and (time.monotonic()-started)/3600>args.max_hours)) if main else 0.0)[0]);last=step+1==steps or out_of_time or (args.max_steps and step+1-step0>=args.max_steps)
 if (step+1)%args.dev_every==0 or last:
  score=evaluate_dev(entry)
  if score<best:best=score;save('best',step+1,best);entry['saved_best']=True
  save('last',step+1,best);seen=0;window=time.monotonic()
 elif args.save_every and (step+1)%args.save_every==0:save('last',step+1,best)
 if main:
  with log.open('a') as f:f.write(json.dumps(entry)+'\n')
 if (step+1)%10==0 or 'dev_loss' in entry:say(json.dumps(entry))
 if last and step+1!=steps:say('Stopped early at step',step+1,'(cost cap or --max-steps)');break
say('Finished. Best selection score ('+args.select+')',best)
if world>1:dist.barrier();dist.destroy_process_group()
