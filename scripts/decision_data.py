"""Shared loading and rendering for decision-v0 training and evaluation: one code path, so train == serve."""
import json,math,random,hashlib
from collections import namedtuple
from pathlib import Path
from PIL import Image
from vision_decision.contracts import Request,UNKNOWN
from vision_decision.scoring import compile_question
from vision_decision.jev_api import flatten_instructions

VERSION='decision-v0'

# load_records splits on '\\n', not splitlines(): U+2028 inside JSON strings would otherwise cut a row (phase 2c)
def load_records(partition=None,version=VERSION):
 rows=[json.loads(l) for l in Path(f'data/manifests/{version}.jsonl').read_text().split('\n') if l.strip()];index=Path(f'data/manifests/{version}-images.json')
 if index.exists():
  images=json.loads(index.read_text())['images'];rows=[dict(r,**images[r['id']]) for r in rows]
 for r in rows:r.setdefault('images',[dict(image=r['image'],sha256=r['sha256'])] if 'image' in r else [])
 rows=[r for r in rows if partition is None or r['partition']==partition]
 return [item for r in rows for item in expand_fields(r)]

def expand_fields(record):
 """Expand a multi-question request into one training item per field.

 Converters may store ``targets`` and ``target_distributions`` keyed by field id.
 Single-question v0/v1 records remain byte-for-byte compatible apart from a defensive
 deep copy when expansion is needed.
 """
 fields=record['request']['fields']
 targets=record.get('targets',{});distributions=record.get('target_distributions',{})
 if len(fields)==1 and not targets and not distributions:return [record]
 out=[]
 for field in fields:
  copy=json.loads(json.dumps(record));copy['id']=f"{record['id']}:{field['id']}";copy['request']['fields']=[field]
  if field['id'] not in targets:raise ValueError(f"{record['id']}: missing target for field {field['id']}")
  copy['target']=targets[field['id']]
  copy['abstention_cause']=(record.get('abstention_causes') or {}).get(field['id'])
  if field['id'] in distributions:copy['target_distribution']=distributions[field['id']]
  copy.pop('targets',None);copy.pop('target_distributions',None);copy.pop('abstention_causes',None)
  out.append(copy)
 return out

TEXT_RICH={'textvqa','dude','heldout_infovqa'}  # text must stay legible
def pixel_budget(record,default):return max(default,1_000_000) if record.get('source') in TEXT_RICH else default

def load_image(path,pixels):
 image=Image.open(path).convert('RGB');scale=min(1,math.sqrt(pixels/(image.width*image.height)))
 return image.resize((max(1,int(image.width*scale)),max(1,int(image.height*scale))),Image.Resampling.LANCZOS) if scale<1 else image

# Opt-in soft target (--soft-targets): `gold` is the index dev accuracy scores against (the record's
# `target` when present, else the argmax of `target_probs`); `probs` is the distribution over the
# rendered choices in their rendered order, unknown last.
SoftTarget=namedtuple('SoftTarget','gold probs')

def distribution_weights(record,choices,soft):
 """Normalized weights of a label->probability dict over the rendered choices (same order).

 Unknown is a trainable outcome too. Accepts the reserved key and the readable aliases emitted by
 older typed-decision exports."""
 aliases={UNKNOWN:{UNKNOWN,'unknown','null'}}
 for value,_ in choices:aliases.setdefault(value,{str(value),str(value).lower()} if isinstance(value,bool) else {str(value)})
 recognized=set().union(*aliases.values())
 unknown_keys=set(soft)-recognized
 if unknown_keys:raise ValueError(f"{record['id']}: target distribution has unknown keys {sorted(unknown_keys)}")
 def weight(value):
  found=[float(soft[k]) for k in aliases[value] if k in soft]
  if len(found)>1:raise ValueError(f"{record['id']}: duplicate aliases in target distribution for {value}")
  return found[0] if found else 0.0
 weights=[weight(value) for value,_ in choices];total=sum(weights)
 if not all(math.isfinite(w) and w>=0 for w in weights) or total<=0:raise ValueError(f"{record['id']}: invalid target distribution")
 return [w/total for w in weights]

def resolve_probs_target(record):
 """--soft-targets: a record carrying `target_probs` but no `target` gets its gold from the argmax
 (first maximum in the record's own option order). Returns the record unchanged when it has a target."""
 if 'target' in record or not record.get('target_probs'):return record
 parsed=Request.model_validate(record['request']);_,choices,_=compile_question(parsed.fields[0],parsed.state)
 weights=distribution_weights(record,choices,record['target_probs']);value=choices[max(range(len(weights)),key=weights.__getitem__)][0]
 return dict(record,target=None if value==UNKNOWN else value,abstention_cause=record.get('abstention_cause'))

def permute_options(record,seed,epoch):
 """--permute-options: shuffle a choice question's options, deterministically from seed+epoch+record id.

 Targets are stored as option values (and `target_probs`/`target_distribution` are keyed by value),
 so render() re-derives the target index and the soft vector in the new order: nothing to remap by hand.
 Ordinal levels are an ordered scale and boolean yes/no is a fixed pair, so both are returned as is;
 unknown is appended by compile_question and therefore always stays last, as served."""
 field=record['request']['fields'][0]
 if field['type']!='choice' or len(field['options'])<2:return record
 copy=json.loads(json.dumps(record));random.Random(f"{seed}:{epoch}:{record['id']}:options").shuffle(copy['request']['fields'][0]['options'])
 return copy

def render(record,rng=None,shuffle=True,soft_targets=False):
 """(header, choices, texts, target). With rng (and shuffle), choice options are reshuffled; unknown stays last as served.

 target is the gold index, or a list (normalized `target_distribution`), or with soft_targets and a
 `target_probs` dict a SoftTarget(gold index, probs)."""
 request=json.loads(json.dumps(record['request']));field=request['fields'][0]
 if rng is not None and shuffle and field['type']=='choice':rng.shuffle(field['options'])
 parsed=Request.model_validate(request);header,choices,texts=compile_question(parsed.fields[0],parsed.state)
 if soft_targets:record=resolve_probs_target(record)
 wanted=UNKNOWN if record['target'] is None else record['target']
 index=[i for i,(value,_) in enumerate(choices) if value==wanted and type(value)==type(wanted)]
 assert len(index)==1,record['id']
 if soft_targets and record.get('target_probs'):
  return header,choices,texts,SoftTarget(index[0],distribution_weights(record,choices,record['target_probs']))
 soft=record.get('target_distribution')
 if soft:  # e.g. rating histograms: train toward the distribution, not just its mode
  return header,choices,texts,distribution_weights(record,choices,soft)
 return header,choices,texts,index[0]

RATIONALE_PREFIX=' Because: '
def rationale_text(record):
 """The auxiliary rationale continuation for --rationale-weight, or None."""
 r=record.get('rationale')
 return RATIONALE_PREFIX+r.strip() if isinstance(r,str) and r.strip() else None

def approx_rationale_tokens(record,cap):
 """Token-budget estimate for the appended rationale (same ~3.2 chars/token rule as approx_tokens), capped."""
 text=rationale_text(record) if cap>0 else None
 return min(cap,int(len(text)/3.2)+1) if text else 0

def approx_tokens(record,default_pixels,rationale_tokens=0):
 """Cheap length estimate for bucketing: one visual token per 32x32 pixels after the budget, plus text
 (plus the capped rationale continuation when --rationale-weight appends one; rationale_tokens is the cap)."""
 budget=pixel_budget(record,default_pixels);visual=sum(min(im.get('width',640)*im.get('height',480),budget)//1024 for im in record['images'])
 field=record['request']['fields'][0];text=len(field['question'])+sum(len(str(o.get('value','')))+len(str(o.get('description') or ''))+4 for o in field.get('options',field.get('levels',[])))+len(json.dumps(record['request'].get('state',{}),ensure_ascii=False))
 return int(visual+text/3.2+90)+approx_rationale_tokens(record,rationale_tokens)

def batch_plan(records,default_pixels,token_budget,max_batch,seed,epoch,chunk=4096,rationale_tokens=0):
 """Deterministic batches of similar-length examples: padded size (longest x count) stays under token_budget."""
 order=list(range(len(records)));random.Random(f'{seed}:{epoch}').shuffle(order);batches=[]
 for start in range(0,len(order),chunk):
  part=sorted(order[start:start+chunk],key=lambda i:approx_tokens(records[i],default_pixels,rationale_tokens));current=[];longest=0
  for i in part:
   n=approx_tokens(records[i],default_pixels,rationale_tokens)
   if current and (max(longest,n)*(len(current)+1)>token_budget or len(current)>=max_batch):batches.append(current);current=[];longest=0
   current.append(i);longest=max(longest,n)
  if current:batches.append(current)
 random.Random(f'{seed}:{epoch}:batches').shuffle(batches);return batches

NOISE_STATES=[{"request_source":"mobile_app","locale":"en-US"},{"trace_id":"a81f","priority":"normal"},{"client":{"name":"web","version":"4.2.1"}},{"batch":17,"operator":"queue-3"},
 {"timestamp":"2026-03-14T09:21:07Z","region":"eu-west"},{"session":{"id":"s-20391","retry":0}},{"workflow":"intake","step":3},{"tenant":"acme","dry_run":False}]
NO_MATCH=['other','none of the above','none of these','something else','not listed','no match']
def with_noise_state(record,rng,rate=.15,object_rate=.06,no_match_rate=.35,decoy_rate=.10):
 """Request-style augmentation that never changes what is true about the image.

 - irrelevant state: real requests carry state unrelated to a given question; empty state must not be all the model has seen;
 - object-form instructions: Jev accepts {"question": "..."}; served text is the flattened "question: ..." line;
 - user-supplied no-match option: Jev has no built-in abstention and tells users to add an "other"/"none of the above" option.
   When the right answer is not listed AND the user offers such an option, that option is the correct answer, not "unknown";
   the same option is sometimes added to answerable questions as a wrong choice so its presence gives nothing away.
 """
 field=record['request']['fields'][0];noise=not record['request'].get('state') and rng.random()<rate
 wrap=rng.random()<object_rate and '\n' not in field['question'] and not field['question'].startswith('question: ')
 room=field['type']=='choice' and len(field['options'])<25 and not any(str(o['value']).strip().lower() in NO_MATCH for o in field['options'])
 no_match=room and ((record['abstention_cause']=='not_listed' and rng.random()<no_match_rate) or (record['target'] is not None and rng.random()<decoy_rate))
 if not (noise or wrap or no_match):return record
 copy=json.loads(json.dumps(record));field=copy['request']['fields'][0]
 if noise:copy['request']['state']=rng.choice(NOISE_STATES)
 if wrap:field['question']=flatten_instructions({'question':field['question']})
 if no_match:
  label=rng.choice(NO_MATCH);field['options'].append({'value':label})
  if copy['abstention_cause']=='not_listed':copy['target']=label;copy['abstention_cause']=None;copy.pop('target_probs',None)  # the probs describe the unaugmented options
 return copy
