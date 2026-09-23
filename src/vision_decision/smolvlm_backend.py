"""Research-only SmolVLM2 adapter; preserves the Qwen CLI backend."""
import hashlib
from time import perf_counter
from .backend import MLXDirect
from .scoring import verified_label_ids, result_from_logits

class SmolVLMDirect(MLXDirect):
    def score_compiled(self,image,prompt,labels,choices):
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens
        mx=self.mx;start=perf_counter()
        if self.model.config.model_type!='smolvlm':raise ValueError('Expected SmolVLM2')
        images = [] if image is None else image if isinstance(image, list) else [image]
        if len(images) > 2: raise ValueError('Supports at most two images')
        rendered=apply_chat_template(self.processor,self.model.config,prompt,num_images=len(images),enable_thinking=False)
        boundary='Assistant:'
        if not rendered.endswith(boundary):raise ValueError('Unverified SmolVLM answer boundary')
        tokenizer=self.processor.tokenizer
        token_ids=verified_label_ids(tokenizer,rendered,labels)
        inputs=prepare_inputs(self.processor,images=images or None,prompts=rendered,
            image_token_index=getattr(self.model.config,'image_token_index',None),
            add_special_tokens=should_add_special_tokens(self.model.config.model_type,self.processor))
        ids=inputs.pop('input_ids');pixels=inputs.pop('pixel_values', None);mask=inputs.pop('attention_mask',None)
        suffix=tokenizer.encode(boundary,add_special_tokens=False)
        if ids[0,-len(suffix):].tolist()!=suffix:raise ValueError('SmolVLM processed suffix mismatch')
        if ids.shape[0]!=1 or ids.shape[-1]>4096:raise ValueError('Unsupported sequence shape')
        if mask is not None and not bool(mx.all(mask==1).item()):raise ValueError('Padding unsupported')
        preprocess=perf_counter()-start;start=perf_counter()
        # The processor mask is a 2-D padding mask, not a causal attention mask.
        # With no padding, let SmolVLM construct its native causal masks.
        output=self.model(ids,pixels,mask=None,logits_to_keep=1,**inputs)
        selected=output.logits[0,-1,mx.array(token_ids)].astype(mx.float32)
        mx.eval(selected);logits=selected.tolist()
        metadata=dict(preprocess_seconds=preprocess,forward_seconds=perf_counter()-start,input_tokens=ids.shape[-1],
            choice_token_ids=token_ids,template_sha256=hashlib.sha256(rendered.encode()).hexdigest(),template_suffix=rendered[-64:],
            peak_memory_bytes=mx.get_peak_memory(),logical_forward_evaluations=1,
            image_grid_thw=None,image_count=len(images),pixel_values_shape=([list(x.shape) for x in pixels] if isinstance(pixels,list) else list(pixels.shape)) if pixels is not None else None,
            visual_tokens=int(mx.sum(ids==self.model.config.image_token_index).item()),
            tie_break_policy='lowest_vocabulary_token_id',attention_mask_policy='native causal masks; verified unpadded input')
        return result_from_logits(choices,logits,token_ids=token_ids),metadata
